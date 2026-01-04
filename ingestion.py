import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Configuration
DATA_PATH = "student_data/"
DB_PATH = "chroma_db_student"

def create_student_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Created '{DATA_PATH}'. Add Student & Career PDFs here.")
        return

    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_docs = loader.load()
    
    if len(raw_docs) == 0:
        print("⚠️ No PDFs found in student_data folder!")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(raw_docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(documents, embeddings, persist_directory=DB_PATH)
    print(f"✅ Student Knowledge Base Ready!")

if __name__ == "__main__":
    create_student_db()