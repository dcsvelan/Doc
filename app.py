import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the Streamlit app
st.title("🤖 Simple AI Agent")

# Get user input
context = st.text_area("Enter the context:")
question = st.text_input("Enter your question:")

# Create a button to generate the answer
if st.button("Get Answer"):
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