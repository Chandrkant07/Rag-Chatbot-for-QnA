# RAG Chatbot: Document Q&A using LLMs

A Retrieval-Augmented Generation (RAG) chatbot that allows users to have natural language conversations with their PDF documents.

## Features
* **PDF Processing:** Upload any PDF and the app will extract and process the text.
* **Semantic Search:** Uses Hugging Face embeddings (`all-MiniLM-L6-v2`) and FAISS to create a vector database for extremely fast and relevant similarity search.
* **Conversational AI:** Integrates with OpenAI's LLM to provide accurate, context-aware answers based *only* on the content of the uploaded document.
* **Chat Memory:** Remembers the context of the conversation so you can ask follow-up questions naturally.

## Tech Stack
* **Python**
* **Streamlit** (Frontend)
* **LangChain** (Orchestration)
* **Hugging Face Transformers** (Embeddings)
* **FAISS** (Vector Database)
* **OpenAI API** (LLM generation)
* **PyPDF** (Document parsing)

## Setup Instructions

1.  **Clone or download the repository.**
2.  **Ensure you have Python 3.9+ installed.**
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    *   Open the `.env` file (or create one in the root directory).
    *   Add your OpenAI API key:
        ```env
        OPENAI_API_KEY="your-actual-api-key"
        ```
5.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

---

## 👨‍💻 Interview Explanation Guide

*Use this section to explain your project to an interviewer.*

**1. "Can you explain what this project is and why you built it?"**
> "I built a RAG (Retrieval-Augmented Generation) chatbot to solve the problem of manual document search. Instead of Ctrl+F-ing through large PDFs, users can upload a document and ask natural language questions. It reduces search time by summarizing and finding exact context instantly."

**2. "Walk me through the architecture. How does it work end-to-end?"**
> "The pipeline has two main phases: Ingestion and Retrieval/Generation.
> *   **Ingestion:** When a user uploads a PDF, I use `PyPDF` to extract the text. Since LLMs have context limits, I split the text into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`. Then, I pass these chunks through a local Hugging Face embedding model (`all-MiniLM-L6-v2`) to turn the text into high-dimensional vectors. I store these vectors in a FAISS vector database.
> *   **Retrieval & Generation:** When the user asks a question, the system embeds their question using the same Hugging Face model. It then performs a similarity search in FAISS to find the top most relevant chunks from the PDF. Finally, I pass the user's question, the retrieved context chunks, and the chat history into OpenAI's LLM via a LangChain `ConversationalRetrievalChain`. The LLM then generates an accurate answer based purely on that context."

**3. "Why did you choose FAISS and Hugging Face instead of Pinecone or OpenAI embeddings?"**
> "I chose FAISS because it runs locally in memory, making it incredibly fast and cost-effective for individual document Q&A without needing a cloud database like Pinecone. I chose Hugging Face's `MiniLM` for embeddings because it provides excellent semantic representation while being small enough to run efficiently on a CPU, which saves costs compared to hitting an API for embeddings every time. I saved the API calls specifically for the final reasoning step using OpenAI."

**4. "How do you handle context or follow-up questions?"**
> "I implemented a memory buffer using Streamlit's session state and LangChain's memory modules. The system keeps a history of the chat. When a follow-up question is asked, the chain first uses the LLM to rephrase the question into a standalone query (taking chat history into account), and then searches the vector database with that standalone query."
