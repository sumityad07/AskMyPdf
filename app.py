import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import re

# Load model
@st.cache_resource
def load_model():
    model_name = "declare-lab/flan-alpaca-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_model()

# Prompt templates
question_prompt = PromptTemplate.from_template(
    '''
You are a highly experienced subject matter expert and interviewer.

Your task is to read the following document excerpt and generate 8 unique and insightful questions that could be used in an interview to assess a candidate's understanding of the material.

Here is the document excerpt:
{text}
'''
)

answer_prompt = PromptTemplate.from_template(
    '''
You are a highly skilled subject matter expert.

Answer the following question in a detailed, technical, and professional manner.

Question:
{text}
'''
)

question_chain = LLMChain(llm=llm, prompt=question_prompt)
answer_chain = LLMChain(llm=llm, prompt=answer_prompt)

# Streamlit UI
st.title("📄 PDF Interview Q&A Generator with AI")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n\n".join([doc.page_content for doc in pages])

    if st.button("🔍 Generate Questions and Answers"):
        with st.spinner("Generating questions..."):
            q_output = question_chain.invoke({"text": full_text})
            questions_text = q_output['text']
            st.subheader("📌 Generated Questions")
            st.write(questions_text)

            # Split questions
            questions = re.findall(r"\d+\.\s+(.*?)(?=\d+\.|$)", questions_text, re.DOTALL)

            st.subheader("🧠 Answers")
            for i, q in enumerate(questions, 1):
                answer = answer_chain.invoke({"text": q})['text']
                st.markdown(f"**Q{i}. {q.strip()}**")
                st.markdown(f"{answer.strip()}")
