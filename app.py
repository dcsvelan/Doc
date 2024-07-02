import streamlit as st
from transformers import pipeline

# Set up the Streamlit app
st.title("🤖 Simple AI Agent (Using Hugging Face)")

# Initialize the language model
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

model = load_model()

# Get user input
user_input = st.text_input("Enter your question:")

# Create a button to generate the answer
if st.button("Get Answer"):
    try:
        # Generate the answer
        answer = model(user_input, max_length=50)[0]['generated_text']

        # Display the answer
        st.write("Answer:", answer)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.info("This AI agent uses the FLAN-T5 model from Hugging Face, which is free to use and doesn't require an API key.")