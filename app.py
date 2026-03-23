import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Load env
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="PDF Chat (Groq)", layout="wide")
st.title("📄 Chat with your PDF (Groq + LLaMA3)")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
retrieval_chain = None

# Load API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Force model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    groq_api_key=groq_api_key
)

if uploaded_file:
    # Save PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB (Chroma)
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectordb.as_retriever()

    # LLM (Groq)
    llm = ChatGroq(
        model=model_choice,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the context below.
        If you don't know, say you don't know.

        Context:
        {context}

        Question:
        {input}
        """
    )

    # Chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    st.success("PDF processed successfully; you can now ask questions.")

if not uploaded_file:
    st.info("Upload a PDF first to activate the question input.")

query = st.text_input("Ask a question about your PDF")

if query:
    if retrieval_chain is None:
        st.warning("Please upload and process a PDF before asking a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                response = retrieval_chain.invoke({"input": query})
                st.write("### Answer:")
                st.write(response.get("answer", "No answer returned"))
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.info("Try a different model in the dropdown (e.g., groq-7b, llama3-16b, llama3-70b).")