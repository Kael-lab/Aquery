import os
import fitz # PyMuPDF
import streamlit as st
from dotenv import load_dotenv  
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

load_dotenv()
HF_API_KEY= os.getenv("HF_API_KEY")

@st.cache_resource
def load_llm():
    pipe=pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizers="google/flan-t5-base",
        device=-1,
        max_new_tokens=512,
    )
    return HuggingFacePipeline(pipeline=pipe)
llm=load_llm()

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=""" You are a helpful AI assistant. Use the context below to answer the user's question.
Context:
{context}
Question: 
{question}

Answer:"""
)
def exract_text_from_pdf(uploaded_file):
    doc=fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text=""
    for page in doc:
        text+=page.get_text()
    return text

st.set_page_config(page_title="Langchain PDF Chat", layout= "centered")
st.title("LanghChain PDF Chat (RAG)")
uploaded_file=st.file_uploader("Upload a PDF",type=["pdf"])
if uploaded_file:
    with st.spinner("Reading and splitting your PDF..."):
        raw_text=exract_text_from_pdf(uploaded_file)

        splitter =RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
chunks=splitter.split_text(raw_text)
embedder=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb=FAISS.from_texts(chunks, embedding=embedder)

retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})

qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={"prompt": QA_PROMPT}
)
st.success("PDF processed! Ask away")
st.session_state.qa_chain=qa_chain

if "qa_chain" in st.session_state:
    question = st.text_input("Ask a question about your PDF:")

if question:
    with st.spinner("Thinking..."):
        try:
            result=st.session_state.qa_chain.run(question)
            st.markdown("**Answer:**")
            st.write(result)
        except Exception as e:
            st.error(f"Error: {str(e)}")

                   



