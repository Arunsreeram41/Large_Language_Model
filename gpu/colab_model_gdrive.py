# Importing the necessary packages
import time
import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
import torch
from torch import cuda
from langchain.chains import RetrievalQA

model_path= '/content/drive/MyDrive/Colab Notebooks/model/mistral-7b-claude-chat.Q4_K_M.gguf'
pdf_folder = 'pdfs'
chunk_size = 1000
chunk_overlap = 20

#function to load the llm model
def load_llm(model_path):
    llm = LlamaCpp(model_path=model_path,
                   temperature=0.1,
                   n_gpu_layers=50,
                   n_ctx=2048)
    return llm


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
def split_docs(docs, chunk_size, chunk_overlap):

  # Initializing the RecursiveCharacterTextSplitter with
  # chunk_size and chunk_overlap
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  chunks = text_splitter.split_documents(docs)
  return chunks



# Function for loading and creating embeddings using FAISS
def create_embeddings(chunks):
  device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

  if device !='cpu':
    model_kwargs = {"device": "cuda"}
  else:
    model_kwargs = {"device": "cpu"}
    
  model_name = "sentence-transformers/all-mpnet-base-v2"

  embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

  # storing embeddings in the vector store
  vectorstore = FAISS.from_documents(chunks, embeddings)
  return vectorstore


# Creating the chain for Question Answering
# Creating the chain for Question Answering
def load_qa_chain(retriever, llm):
  return RetrievalQA.from_chain_type(
      llm=llm,
      retriever=retriever, # here we are using the vectorstore as a retriever
      chain_type="stuff",
      return_source_documents=True # including source documents in output
  )


def get_response(query, chain):
  # Getting response from chain
  start_time=time.time()
  response = chain({'query': query})
  end_time=time.time()
  duration=(end_time-start_time)/60
  return response['result'],response['source_documents'][0],duration



#loading the llm model
llm = load_llm(model_path)

# loading and splitting the documents
docs,pdf_files,n_pdf_files,n_pages = load_pdf_data(pdf_folder)
chunks = split_docs(docs,chunk_size,chunk_overlap)


# creating vectorstore
vectorstore = create_embeddings(chunks)

# converting vectorstore to a retriever
retriever = vectorstore.as_retriever()


# Creating the chain
chain = load_qa_chain(retriever, llm)