from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc_loader = PyPDFLoader(file_path="pdfs/llm_paper.pdf")
documents = doc_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
vector_store.save_local("faiss_index")
print("Index built and saved.")