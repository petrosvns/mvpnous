from main import ChatBot
import streamlit as st

bot = ChatBot()

st.set_page_config(page_title="Symptom Chatbot")

with st.sidebar:
    st.title('Hi there! I am a mental health symptom analyzing chatbot!')

def conv_past(inp):
    ret = []
    for num, comb in enumerate(inp):
        ret.append(f"Message {num%2} by the {comb['role']}: {comb['content']}\n")
    return ret

def generate_response(input_text):
    result = bot.rag_chain.invoke({
        "context": bot.docsearch.as_retriever(),
        "question": input_text,
        "pasts": str(conv_past(st.session_state.messages))
    })
    return result

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi there! How can I help?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if input_text := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.write(input_text)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = generate_response(input_text)
                st.write(response)

        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
