# Large_Language_Model
## :dart:Description :

The aim of this project is to build a pdf chatbot using open source Large Language Models without data exposure (private LLM)

## :crystal_ball:Libraries :
* langchain
* langchain-community
* llama-cpp
* faiss
* sentence-transformers
* pypdf
* streamlit

## :gem:Models :
* Mistral claude chat (7 billion parameters) - mistral-7b-claude-chat.Q4_K_M.gguf
  
  link to the model : https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GGUF/blob/main/mistral-7b-claude-chat.Q4_K_M.gguf
  
  model size : 4.37 GB

* Stable LM zephyr (3 billion parameters) - stablelm-zephyr:3b-q6_K
  
  model source : ollama
  
  link to the model :  https://ollama.com/library/stablelm-zephyr:3b-q6_K
  
  model size : 2.3 GB

## :computer:Hardware and software requirements and response times:
* The Mistral claude chat model was run in GPU (T4 - google colab)

  environment : google colab

  response time : 20 - 30 secs (variable)

* The Stable LM zephyr model was run in CPU (8GB RAM)

  environment : WSL (Windows Subsytem for Linux)

  response time : 6 - 7 mins (variable)

## :book:steps to run:
# GPU folder -
  
  * download the model from the hugging face repository
  * upload the model in the google drive
  * upload the pdfs folder containing pdfs in the colab environment
  * upload the python file (colab_model_gdrive.py) which contains the LLM code
  * run the ipynb script

#  CPU folder - 

This requires Ollama. Ollama allows you to run open-source large language models, such as Llama 2, locally.




    

  

  

  
  

