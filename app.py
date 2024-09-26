import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from FlagEmbedding import FlagModel
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import numpy as np
from typing import List
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from groq import Groq
from langchain_groq import ChatGroq
import time
from tenacity import retry, stop_after_attempt, wait_exponential


class FlagModelAdapter(Embeddings):
    def __init__(self, model_name: str):
        self.model = FlagModel(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

def get_vector_store(text_chunk):
    #Embedding Model
    model = FlagModelAdapter('BAAI/bge-m3')
    v_s_ = FAISS.from_texts(text_chunk, model)
    return v_s_

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key="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",
        model_name = "llama-3.1-70b-versatile",
        temperature=0,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

    return conversation_chain

def get_pdf_text(pdf_docs):
    text = ""
    for doc in pdf_docs:
        pdf = PdfReader(doc)
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_text_loader(pdf):
    text = ""
    pdf = PyPDFLoader(pdf)
    pages = pdf.load()
    for page in pages: 
        text += page.page_content + "\n\n"
    return text

def get_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1250,
        chunk_overlap=100,
        separators = ["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(docs)

def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.columns(2)[1]:
                st.markdown(
                f"""
                <div style='background-color: #f1f1f1; padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 10px 0; color: #333;'>
                    <strong style="color: #333;">Pregunta:</strong> {message.content}
                </div>
                """,
                unsafe_allow_html=True
                )
                
        else:
            with st.columns(2)[0]:
                st.markdown(
                f"""
                <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; border: 1px solid #90caf9; margin: 10px 0; color: #0d47a1;'>
                    <strong style="color: #0d47a1;">Respuesta:</strong> {message.content}
                </div>
                """,
                unsafe_allow_html=True
                )

def clear_chat():
    st.session_state.chat_history.clear()

def main():
    load_dotenv()
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDF's :books:")
    user_question = st.chat_input("Ask a question about your documents:")
    
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF's", accept_multiple_files=True)
        if st.button("process"):
            #Extract the text from the documents
            raw_text = get_pdf_text(pdf_docs)

            #get the chunks 
            chunks = get_chunks(raw_text)
            
            #Vector Store
            VStore = get_vector_store(chunks)

            #Create conversation chain
            st.session_state.conversation = get_conversation_chain(VStore)

            st.success("Documents processed successfully")

        if st.button("Clear Chat"):
            clear_chat()
            st.success("Chat history cleared!")
            
if __name__ == "__main__":
    main()
