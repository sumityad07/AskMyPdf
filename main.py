
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.schema import Document
import torch
from langchain.embeddings import HuggingFaceEmbeddings
# ✅ GPU Check
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

# ✅ Load PDF
loader = PyPDFLoader("data/research paper.pdf")
pages = loader.load()
full_text = "\n\n".join([doc.page_content for doc in pages])

# ✅ Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.create_documents([full_text])

# ✅ Load lightweight instruction-tuned model
model_name = "declare-lab/flan-alpaca-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ✅ Create pipeline
text_gen_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text2text-generation",
    device=0 if torch.cuda.is_available() else -1,
    max_length=512,
    pad_token_id=tokenizer.eos_token_id
)

# ✅ Wrap with LangChain
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# ✅ Prompt Template
prompt_template = PromptTemplate.from_template(
    '''
You are a highly experienced subject matter expert and interviewer.

Your task is to read the following document excerpt and generate 8 unique and insightful questions that could be used in an interview to assess a candidate's understanding of the material.

Ensure the questions:
- Are clear, concise, and professional
- Test real-world application and deep understanding
- Avoid repetition or surface-level details

Here is the document excerpt:
{text}
'''
)

refine_prompt = PromptTemplate.from_template(
    '''
We have already generated the following interview questions:

{existing_answer}

Now, using the additional content provided below, refine the questions to improve their clarity, technical depth, or relevance. Add 1–2 new questions only if the new information introduces important new topics.

Additional content:
{text}
'''
)

ques_gen_chain = load_summarize_chain(
     llm=llm,
    chain_type="refine",
    question_prompt=prompt_template,   # 👈 Initial chunk handler
    refine_prompt=refine_prompt,      # 👈 Refine logic for additional chunks
    
)

# Generate 5 questions
questions = ques_gen_chain.invoke(chunks)



# Convert string questions to a LangChain Document
questions_text = questions['output_text'] if isinstance(questions, dict) else str(questions)
question_doc = [Document(page_content=questions_text)]


from langchain_core.output_parsers import StrOutputParser

import re
questions_list = re.findall(r"\d+\.\s+(.*?)(?=\d+\.|$)", questions_text, re.DOTALL)

# ✅ Create QA prompt
qa_prompt = PromptTemplate.from_template("""
You are a highly knowledgeable AI assistant.

Answer the following question in detail based on your understanding of the topic:

Question: {question}
""")

# ✅ Use the prompt + model
qa_chain = qa_prompt | llm | StrOutputParser()

# ✅ Ask each question
for i, q in enumerate(questions_list, 1):
    question = q.strip()
    if not question:
        continue
    print(f"\n📌 Question {i}: {question}")
    answer = qa_chain.invoke({"question": question})
    print(f"🧠 Answer {i}: {answer}")