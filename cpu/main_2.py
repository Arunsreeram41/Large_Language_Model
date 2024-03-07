import llm_pdf
from llm_pdf import chain,pdf_files,n_pdf_files,get_response
import streamlit as st
import time




#streamlit
st.title("LLM chatbot")
st.subheader("chat with your pdfs")
st.text("")
st.text("")
st.write("Pdf files present : ")
for pdf in pdf_files:
    st.write(pdf)
st.write("No.of pdfs : ",n_pdf_files)
# st.write("Total no.of pages : ",n_pages)
st.text("")

#initialise sessoin_state: not to make the entire application to rerun,everytime the user interacts
if 'load_state' not in st.session_state:
    st.session_state.load_state=False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages=[] #there is no attribute called messages for session_state, we only create something called messages here


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
query = st.chat_input("Enter your query")
if query or st.session_state.load_state:
    st.session_state.load_state=True
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role":"user","content":query})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        placeholder = st.empty()
        model_response = ""
        responses,source_doc,duration = get_response(query,chain)

        for response in responses.split():
            model_response+=response+" "
            time.sleep(0.05)
            placeholder.markdown(model_response+"▌")
        placeholder.markdown(model_response)
        st.text("")
        st.write("Duration of response : ",duration,"mins")
    st.session_state.messages.append({"role":"assistant","content":model_response})


#responses ="hi how r u"
#model_response = "hi▌"
#model_response = "hi how▌"   
#model_response = "hi how r u▌"