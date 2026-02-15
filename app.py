import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdfextractor import text_extractor_pdf 

from pypdf import PdfReader
import streamlit as st  

from dotenv import load_dotenv  

load_dotenv()

#Create the main page of the Streamlit app

st.title(':green[RAG Application Demo]')

#load the PDF file and extract text (from the sidebar)
st.sidebar.header('Upload your PDF file')
file_uploaded = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)

    #Step 1: Configure the model
    key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=key)
    llm_model = genai.GenerativeModel('gemini-3-flash-preview')

    #Step 2: Configure the embeddings
    embedding_model = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2") 

    #Step 3: Split the text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    #step 4: Create the vector store
    vector_store = FAISS.from_texts(chunks, embedding_model)

    #step 5: Configure teh retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    #Function to generate response from the model
    def generate_response(query: str) -> str:
        retrieved_docs=retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"""
        You are a helpful assistant using RAG. Use the following context to answer the question: {context}
        User Query : {query}
        """
    
        #generation
        content = llm_model.generate_content(prompt)
        return content.text if hasattr(content, 'text') else content.candidates[0].content.parts[0].text
    
    #Initialize the chat history
    if 'history' not in st.session_state:
        st.session_state.history = []
        
    # Display the History
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f':green[User:] :blue[{msg["text"]}]')
        else:
            st.write(f':orange[Chatbot:] {msg["text"]}')
    # Input from the user (Using Streamlit Form)
    with st.form('Chat Form', clear_on_submit=True):
        user_input = st.text_input('Enter Your Text Here:')
        send = st.form_submit_button('Send')

    # Start the conversation and append the output and query in history
    if user_input and send:
        st.session_state.history.append({"role": 'user', "text": user_input})
        model_output = generate_response(user_input)
        st.session_state.history.append({'role': 'chatbot', 'text': model_output})
        st.rerun()