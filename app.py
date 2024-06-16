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
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
    
    st.title("🤖 General AI Chatbot")
    st.write("Welcome to the AI Chatbot! Ask any question or start a conversation.")

    # Sidebar for additional options
    st.sidebar.title("Options")
    st.sidebar.write("Adjust the chatbot settings below:")

    # Add a text input for user queries
    user_input = st.text_input("You: ", "")

    # Add a button to submit the query
    if st.button("Send"):
        if user_input:
            llm = load_model()
            with st.spinner("Generating response..."):
                response = generate_response(llm, user_input)
            st.write("**Bot:** " + response)
        else:
            st.warning("Please enter a message to get a response.")

    # Add a footer
    st.markdown("---")
    st.markdown("Developed by [Your Name](https://yourwebsite.com)")

if __name__ == "__main__":
    main()
