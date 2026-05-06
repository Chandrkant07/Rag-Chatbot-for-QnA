import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables (like HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# --- Configuration & UI Setup ---
st.set_page_config(page_title="RAG Chatbot", page_icon="📚")

st.title("📚 Chat with your PDF")
st.markdown("Upload a document and ask questions about its content.")

# --- Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Splits the extracted text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Embeds the text chunks and stores them in a FAISS vector database."""
    # Using a fast, local embedding model from Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Sets up the retrieval chain with Hugging Face and memory."""
    # We use a free open-source model from Hugging Face for answer generation
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.1,
        max_new_tokens=512
    )
    
    # Memory allows the LLM to remember the context of the conversation
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# --- Main Application Flow ---

# Initialize session state variables if they don't exist
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for file upload and processing
with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
    )
    
    if st.button("Process"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF.")
        elif not os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") == "your_huggingface_token_here":
             st.error("Please set your HUGGINGFACEHUB_API_TOKEN in the .env file.")
        else:
            with st.spinner("Processing..."):
                # 1. Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)
                
                # 2. Split text into chunks
                text_chunks = get_text_chunks(raw_text)
                
                # 3. Create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # 4. Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                st.success("Processing complete! You can now ask questions.")

# Chat Interface
for message in st.session_state.chat_history:
    role = "user" if message.__class__.__name__ == "HumanMessage" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Input for user query
user_question = st.chat_input("Ask a question about your documents:")

if user_question:
    if st.session_state.conversation is None:
        st.warning("Please upload and process a document first.")
    else:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)
            
        # Get response from the LLM
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            
            # The last message in chat_history will be the AI's response
            ai_response = st.session_state.chat_history[-1].content
            
            # Display AI response
            with st.chat_message("assistant"):
                st.markdown(ai_response)
