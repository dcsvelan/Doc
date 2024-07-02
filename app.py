import streamlit as st
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up the Streamlit app
st.title("🤖 Simple AI Agent (Using Hugging Face)")

# Initialize the language model
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

model = load_model()
llm = HuggingFacePipeline(pipeline=model)

# Get user input
context = st.text_area("Enter the context:")
question = st.text_input("Enter your question:")

# Create a button to generate the answer
if st.button("Get Answer"):
    try:
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
        st.error("Please try again. If the problem persists, check your internet connection or try a different model.")

st.info("This AI agent uses the FLAN-T5 model from Hugging Face, which is free to use and doesn't require an API key.")