import json
from typing import Any, Dict, List

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

st.set_page_config(
    page_title="QuizGPT",
    page_icon="📚",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "current_questions" not in st.session_state:
    st.session_state["current_questions"] = []

if "correct_count" not in st.session_state:
    st.session_state["correct_count"] = 0


def create_quiz_function():
    return {
        "name": "create_quiz",
        "description": "Create a quiz based on the provided context",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "options": {"type": "array", "items": {"type": "string"}},
                            "correct_answer": {"type": "string"},
                        },
                        "required": ["question", "options", "correct_answer"],
                    },
                }
            },
            "required": ["questions"],
        },
    }


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f".cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def create_quiz(context, difficulty):
    llm = ChatOpenAI(
        temperature=0.7 if difficulty == "Hard" else 0.3,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        openai_api_key=st.session_state.openai_api_key,
        functions=[create_quiz_function()],
        function_call={"name": "create_quiz"},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a professional quiz maker. Create {num_questions} multiple choice questions based on the following context.
        Make the questions {difficulty} difficulty level.
        
        Context: {context}
        """,
            ),
        ]
    )

    chain = prompt | llm

    result = chain.invoke(
        {"context": context, "difficulty": difficulty, "num_questions": 5}
    )

    quiz_data = json.loads(result.additional_kwargs["function_call"]["arguments"])
    return quiz_data["questions"]


# Sidebar
with st.sidebar:
    st.title("Settings")

    # OpenAI API Key input
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        st.session_state.openai_api_key = api_key

    # Difficulty selector
    difficulty = st.selectbox("Select quiz difficulty:", ["Easy", "Hard"])

    # File uploader
    file = st.file_uploader(
        "Upload a file (.txt, .pdf, or .docx)", type=["pdf", "txt", "docx"]
    )

    # Github link
    st.markdown("---")
    st.markdown(
        "[View the code on Github](https://github.com/lution88/lutions-fullstack-gpt)"
    )

# Main content
st.title("📚 QuizGPT")
st.markdown(
    """
Welcome to QuizGPT! Upload a document and I'll create a quiz to test your knowledge.
"""
)

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
elif file:
    retriever = embed_file(file)

    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False

    if not st.session_state.quiz_started:
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz questions..."):
                docs = retriever.invoke(
                    "Generate a comprehensive quiz about the main topics"
                )
                context = format_docs(docs)
                st.session_state.current_questions = create_quiz(context, difficulty)
                st.session_state.quiz_started = True
                st.session_state.current_question = 0
                st.session_state.correct_count = 0
                st.experimental_rerun()

    if st.session_state.quiz_started and st.session_state.current_question < len(
        st.session_state.current_questions
    ):
        question = st.session_state.current_questions[st.session_state.current_question]

        st.subheader(f"Question {st.session_state.current_question + 1}")
        st.write(question["question"])

        answer = st.radio(
            "Choose your answer:",
            question["options"],
            key=f"q_{st.session_state.current_question}",
        )

        if st.button("Submit Answer"):
            if answer == question["correct_answer"]:
                st.success("Correct! 🎉")
                st.session_state.correct_count += 1
            else:
                st.error(f"Wrong! The correct answer was: {question['correct_answer']}")

            st.session_state.current_question += 1
            st.experimental_rerun()

    elif st.session_state.quiz_started:
        st.subheader("Quiz Complete!")
        st.write(
            f"You got {st.session_state.correct_count} out of {len(st.session_state.current_questions)} questions correct!"
        )

        if st.session_state.correct_count == len(st.session_state.current_questions):
            st.balloons()
            st.success("Perfect score! Congratulations! 🎉")
        else:
            if st.button("Try Again"):
                st.session_state.quiz_started = False
                st.experimental_rerun()
else:
    st.info("Please upload a file to begin!")
