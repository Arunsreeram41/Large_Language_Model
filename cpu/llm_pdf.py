# Importing the necessary packages
import time
import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

model = "stablelm-zephyr:3b-q6_K"



# This will load the PDF file
def load_pdf_data(folder_name):
   
    # Creating a PyMuPDFLoader object with file_path
    loader = PyPDFDirectoryLoader(folder_name)
    
    # loading the PDF file
    docs = loader.load()
    pdf_files=os.listdir(folder_name)
    n_pdf_files=len(os.listdir(folder_name))
    n_pages=len(docs)

    
    return docs,pdf_files,n_pdf_files,n_pages
        

# Responsible for splitting the documents into several chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    
    # Initializing the RecursiveCharacterTextSplitter with
    # chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(documents=documents)
    
    # returning the document chunks
    return chunks

# function for loading the embedding model
def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'}, # here we will run the model with CPU only
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding # keep True to compute cosine similarity
        }
    )


# Function for creating embeddings using FAISS
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    # Creating the embeddings using FAISS
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    
    # Saving the model in current directory
    vectorstore.save_local(storing_path)
    
    # returning the vectorstore
    return vectorstore


# Creating the chain for Question Answering
def load_qa_chain(retriever, llm):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True # including source documents in output 
    )

# Prettifying the response
def get_response(query, chain):
    # Getting response from chain
    start_time=time.time()
    response = chain({'query': query})
    end_time=time.time()
    duration=(end_time-start_time)/60
    return response['result'],response['source_documents'][0],duration




# Loading model from Ollama

llm = Ollama(model=model, temperature=0,repeat_penalty=1.5)


# Loading the Embedding Model
embed = load_embedding_model(model_path="all-MiniLM-L6-v2")


# loading and splitting the documents

docs,pdf_files,n_pdf_files,n_pages = load_pdf_data(folder_name="pdfs")
documents = split_docs(documents=docs)


# creating vectorstore
vectorstore = create_embeddings(documents, embed)

# converting vectorstore to a retriever
retriever = vectorstore.as_retriever()

# Creating the chain
chain = load_qa_chain(retriever, llm)


