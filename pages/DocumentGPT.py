import time

import streamlit as st

st.title("Document GPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(msg, role, save=True):
    with st.chat_message(role):
        st.write(msg)
    if save:
        st.session_state["messages"].append({"message": msg, "role": role})


# st.write(st.session_state["messages"])
for message in st.session_state["messages"]:
    send_message(
        message["message"],
        message["role"],
        save=False,
    )


message = st.chat_input("Sned a message to the ai")

if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)
