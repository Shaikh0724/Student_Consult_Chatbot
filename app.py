import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# API Key load karein
load_dotenv()

# --- Premium Purple & Gold UI (Sharp Visibility) ---
st.set_page_config(page_title="Career Guide AI", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    /* Main Background - Deep Royal Purple */
    .stApp { 
        background: linear-gradient(135deg, #2e1065 0%, #1e1b4b 100%); 
        color: #ffffff; 
    }
    
    /* Main Title - Gold & White */
    .main-title { 
        color: #fbbf24; 
        font-size: 42px; 
        font-weight: 800; 
        text-align: center; 
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
    }

    /* Subtitle */
    .sub-title { 
        color: #ddd6fe; 
        text-align: center; 
        font-size: 18px; 
        margin-bottom: 30px;
    }

    /* Chat Messages - Glassmorphism Style with White Text */
    .stChatMessage { 
        background: rgba(255, 255, 255, 0.1) !important; 
        border: 1px solid rgba(124, 58, 237, 0.4);
        border-radius: 15px; 
        padding: 15px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }

    /* Force Pure White Text for Content */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #ffffff !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }

    /* Sidebar and Input Bar */
    .stChatInputContainer {
        background: rgba(46, 16, 101, 0.8) !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>🎓 Student Career & Scholarship AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Expert Roadmap for Admissions, Scholarships & Career Success</div>", unsafe_allow_html=True)

# --- Auto-Ingestion Logic ---
@st.cache_resource
def load_db():
    DB_PATH = "./chroma_db_student"
    if not os.path.exists(DB_PATH):
        with st.spinner("⏳ Organizing academic intelligence base..."):
            from ingestion import create_db # Make sure ingestion.py is in the same folder
            create_db()
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

db = load_db()
retriever = db.as_retriever(search_kwargs={'k': 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# --- UNIVERSAL HYBRID PROMPT (Internal Documents + World Knowledge) ---
template = """You are a highly knowledgeable and encouraging Career Counselor AI.

1. PRIMARY SOURCE: Use the provided Context to answer questions about scholarships, SOPs, admissions, or career paths found in the documents.
2. WORLD KNOWLEDGE: If the question is about current events, world history, geography, science, or general news not in the documents, use your internal training to provide an accurate answer.
3. BE UNIVERSAL: Do not limit yourself to any specific city unless asked. Answer questions about any place in the world.
4. IDENTITY: Mention you are powered by GPT-4o-mini if asked about your model.

Context: {context}
Question: {question}

Helpful Counselor Response:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    if not docs: return "No specific academic documents found for this query."
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# --- Chat Interaction ---
if "student_messages" not in st.session_state:
    st.session_state.student_messages = []

for msg in st.session_state.student_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about scholarships, career paths, or world news..."):
    st.session_state.student_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)
            st.markdown(response)
            st.session_state.student_messages.append({"role": "assistant", "content": response})
            