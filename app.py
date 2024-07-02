import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up the Streamlit app
st.title("🤖 Simple AI Agent")

# Function to get OpenAI API key
def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    return api_key

# Get OpenAI API key
api_key = get_openai_api_key()

if not api_key:
    st.warning("Please enter your OpenAI API key to use the AI Agent.")
else:
    # Get user input
    context = st.text_area("Enter the context:")
    question = st.text_input("Enter your question:")

    # Create a button to generate the answer
    if st.button("Get Answer"):
        try:
            # Initialize the language model
            llm = OpenAI(temperature=0.7)

            # Create a prompt template
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
            )

            # Create an LLMChain
            chain = LLMChain(llm=llm, prompt=prompt)

            # Generate the answer
            answer = chain.run(context=context, question=question)

            # Display the answer
            st.write("Answer:", answer)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your OpenAI API key and try again.")