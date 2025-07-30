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
    template=""" You are a helpful AI assistant.
      Answer the user's question based *only* on the provided context.
Context:
{context}
Question: 
{question}

Answer:"""
)
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
    
    question = st.text_input("What would you like to know from the document(s)?")
    if question:
        with st.spinner("Thinking.."):
            try:
                result = st.session_state.qa_chain.run(question)
                st.write("##Answer")
            except Exception as e:
                st.error(f"An error occurred: {e}")

            else:
                st.info("Please upload and process a document to begin the chat.")

                if __name__ == '__main__':
                    main()
          