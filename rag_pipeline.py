import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # see note below
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def get_answer(query: str) -> str:
    result = vector_store.similarity_search_with_score(query, k=10)
    context = "\n\n".join(doc.page_content for doc, score in result)

    prompt = f"""
Answer the question based only on the context provided below.
Context:
{context}
Question:
{query}
If the answer is not present in the context, say:
"I don't know based on the provided document."
"""
    response = llm.invoke(prompt)
    return response.content