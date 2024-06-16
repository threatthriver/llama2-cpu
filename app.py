import streamlit as st
from ctransformers import AutoModelForCausalLM

# Load the model
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        'TheBloke/Llama-2-7B-Chat-GGML',
        model_file='llama-2-7b-chat.ggmlv3.q4_K_S.bin'
    )
    return model

llm = load_model()

# Streamlit app layout
st.title("General AI Chatbot")
st.write("Ask any question or start a conversation!")

user_input = st.text_input("You: ", "")

if user_input:
    response = ""
    for word in llm(user_input, stream=True):
        response += word
        # Display the response in real-time
        st.write(response, end='')

    st.write("Bot: " + response)
