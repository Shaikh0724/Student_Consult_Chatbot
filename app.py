import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# API Key load karein
load_dotenv()

# --- Configurations ---
DATA_PATH = "student_data/"
DB_PATH = "chroma_db_student"

# --- Automatic Database Creation Function ---
def setup_knowledge_base():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        st.info(f"📁 '{DATA_PATH}' folder created. Please upload PDFs to GitHub.")
        return False

    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_docs = loader.load()
    
    if len(raw_docs) == 0:
        return False

    # Text splitting for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(raw_docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(documents, embeddings, persist_directory=DB_PATH)
    return True

# --- UI Header & Styling ---
st.set_page_config(page_title="Career Guide AI", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #2e1065 0%, #1e1b4b 100%); color: #ffffff; }
    .main-title { color: #fbbf24; font-size: 42px; font-weight: 800; text-align: center; margin-bottom: 0px; }
    .sub-title { color: #ddd6fe; text-align: center; font-size: 18px; margin-bottom: 30px; }
    .stChatMessage { background: rgba(255, 255, 255, 0.1) !important; border: 1px solid rgba(124, 58, 237, 0.4); border-radius: 15px; margin-bottom: 15px; backdrop-filter: blur(10px); }
    .stChatMessage p { color: #ffffff !important; font-size: 16px !important; }
    .stChatInputContainer { background: rgba(46, 16, 101, 0.8) !important; border-radius: 12px !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>🎓 Student Career & Scholarship AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Your Universal Roadmap to Academic & Career Success</div>", unsafe_allow_html=True)

# --- Load Database ---
@st.cache_resource
def get_retriever():
    if not os.path.exists(DB_PATH):
        with st.spinner("⏳ Organizing academic intelligence..."):
            success = setup_knowledge_base()
            if not success:
                return None
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={'k': 4})

retriever = get_retriever()

# --- Chain Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

template = """You are a highly intelligent Career Counselor.
1. PRIORITY: Use Document Context for scholarships, admissions, and career paths.
2. UNIVERSAL KNOWLEDGE: If the answer is not in context, or the user asks about world news, current events, or general facts, provide a detailed answer from your internal knowledge.
3. Be encouraging and clear. Identify as GPT-4o-mini RAG.

Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    if not docs: return "No document context available."
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
if retriever:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
else:
    # Fallback if no PDFs are found
    rag_chain = (
        {"context": lambda x: "No PDFs found.", "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about scholarships or anything in the world..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})