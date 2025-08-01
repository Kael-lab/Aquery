import os
from urllib import response
import fitz # PyMuPDF
import streamlit as st
from dotenv import load_dotenv  
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from torch import chunk
from transformers import pipeline

st.set_page_config(page_title="LangChain PDF Chat", layout="centered"
                   )
load_dotenv()
HF_API_KEY= os.getenv("HF_API_KEY")

def extract_text_from_pdf(pdf_docs):
    """Extract text from a PDF file."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in pdf_reader:
            text += page.get_text()
    return text

@st.cache_resource
def load_llm():
    """Loads the T5-FLAN model for text generation."""
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
    template=""" You are a helpful AI assistant. You specialize in answering questions based on provided context about 
    marine tourism in the Philippines.
    When the user asks for an itenary or travel plan, you will provide a detailed 5-day travel itinerary with specific activities, locations, and recommendations.
    If the user asks for a summary, you will provide a concise summary of the context.
    If the user asks for a specific question, you will answer it based on the context provided.
    If the user asks for a list of activities, you will provide a list of activities based on the context.
    If the user asks for a list of locations, you will provide a list of locations based on the context.
    If the user asks for a list of recommendations, you will provide a list of recommendations based on the context.
    If the user asks for a list of tips, you will provide a list of tips based on the context.
    If the user asks for a list of FAQs, you will provide a list of FAQs based on the context.
    If the user asks for a list of popular destinations, you will provide a list of popular destinations based on the provided context.      
      Answer the user's question based on the provided context.
Context:
{context}
Question: 
{question}

Answer:"""
)
# Streamlit UI
#gradient background
gradient_bg = """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #3151C6, #00003D); 
        background-attachment: fixed;
    }
    </style>
"""
def main():
    st.title("LangChain Chat with PDF (RAG)")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDF(s) here and click 'Process'",
        type="pdf",
        accept_multiple_files=True
    )
if st.button("Process"):

    if pdf_docs:
        with st.spinner("Processing documents..this may take a while"):

            raw_text = extract_text_from_pdf(pdf_docs)
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len 
            )

            chunks=text_splitter.split_text(raw_text)
            embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectordb=FAISS.from_texts(chunks, embeddings)

            llm=load_llm()
            retrieval=vectordb.as_retriever(search_kwargs={'k':5}) # pyright: ignore[reportUndefinedVariable]
            st.session_state.qa_chain=RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retrieval,
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_PROMPT}
            )
        st.success("Processing Complete")
    else:
        st.warning("Please Upload atleast one PDF file")

st.header("Ask a Question")            
if st.session_state.qa_chain:
    
    question = st.text_input("What would you like to know?")
    if question:
        with st.spinner("Thinking.."):
            try:
                result = st.session_state.qa_chain({"query": question})
                st.write(response['result'])

                with st.expander("See source documents"):
                    for doc in response['source_documents']:
                        st.write(doc.page_content)
                        st.caption(f"Source: {doc.metadata['source']}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.info("Please enter a question to get started.")
else:
    if pdf_docs:
        st.info("Processing documents, please wait...")
    else:
        st.info("Please upload and process a document to begin the chat.")

if __name__ == '__main__':
    main()
          