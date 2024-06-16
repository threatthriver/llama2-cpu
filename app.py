import streamlit as st
from ctransformers import AutoModelForCausalLM

# Function to load the model
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        'TheBloke/Llama-2-7B-Chat-GGML',
        model_file='llama-2-7b-chat.ggmlv3.q4_K_S.bin'
    )
    return model

# Function to generate a response from the model
def generate_response(model, user_input):
    response = ""
    for word in model(user_input, stream=True):
        response += word
    return response

# Main function to run the Streamlit app
def main():
    st.title("General AI Chatbot")
    st.write("Ask any question or start a conversation!")

    user_input = st.text_input("You: ", "")

    if user_input:
        llm = load_model()
        response = generate_response(llm, user_input)
        st.write("Bot: " + response)

if __name__ == "__main__":
    main()
