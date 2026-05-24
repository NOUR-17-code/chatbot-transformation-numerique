import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader
import requests
import uuid

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Chatbot Transformation Numérique")
st.markdown("### PDF Assistant avec Groq + Qdrant")

# =========================
# API KEYS
# =========================

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# =========================
# CLIENTS
# =========================

groq_client = Groq(api_key=GROQ_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "rag_collection"

# =========================
# CREATE COLLECTION
# =========================

try:
    qdrant.get_collection(COLLECTION_NAME)

except:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )

# =========================
# EMBEDDING FUNCTION
# =========================

def get_embedding(text):

    response = requests.post(
        "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-m3",
        headers={},
        json={"inputs": text}
    )

    embedding = response.json()

    return embedding[0]

# =========================
# PDF UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type="pdf"
)

if uploaded_file:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    st.success("✅ PDF chargé")

    # =========================
    # CHUNKS
    # =========================

    chunks = []

    chunk_size = 500

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    st.info(f"📚 {len(chunks)} chunks créés")

    # =========================
    # STORE IN QDRANT
    # =========================

    points = []

    for chunk in chunks:

        vector = get_embedding(chunk)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk}
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    st.success("✅ Données stockées dans Qdrant")

    # =========================
    # QUESTION
    # =========================

    question = st.text_input("💬 Posez votre question")

    if question:

        question_vector = get_embedding(question)

        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_vector,
            limit=3
        )

        context = "\n".join(
            [hit.payload["text"] for hit in search_result]
        )

        prompt = f"""
        Réponds à la question en utilisant ce contexte :

        {context}

        Question :
        {question}
        """

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        st.markdown("## 🤖 Réponse")
        st.write(answer)
