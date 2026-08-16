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
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
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
    content = response.content

    # Gemini 3.x returns content as a list of parts instead of a string
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content