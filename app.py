from pathlib import Path

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

st.set_page_config(
    page_title="Assignment #6",
    page_icon="📜",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    Path("./.cache/files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(f"{file_path}")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def main():
    if not openai_api_key:
        return

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        openai_api_key=openai_api_key,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

    memory = ConversationBufferMemory(
        llm=llm,
        max_token_limit=20,
        return_messages=True,
        memory_key="chat_history",
    )

    def load_memory(_):
        return memory.load_memory_variables({})["chat_history"]

    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)

    if file:
        retriever = embed_file(file)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "you are a helpful assistant. answer questions using only the following context. if you don't know the answer just say you don't know, don't make it up: \n{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file.....")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "chat_history": load_memory,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )

            with st.chat_message("ai"):
                result = chain.invoke(message)
                memory.save_context({"input": message}, {"output": result.content})

    else:
        st.session_state["messages"] = []
        return


st.title("Document GPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

1. Input your OpenAI API Key on the sidebar
2. Upload your file on the sidebar.
3. Ask questions related to the document.
"""
)

with st.sidebar:
    # API Key 입력
    openai_api_key = st.text_input("Input your OpenAI API Key")

    # 파일 선택
    file = st.file_uploader(
        "Upload a. txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

try:
    main()
except Exception as e:
    st.error("Check your OpenAI API Key or File")
    st.write(e)
