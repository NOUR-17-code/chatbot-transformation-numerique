import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import uuid

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Chatbot RAG Multilingue")
st.markdown("### 🇫🇷 🇬🇧 🇲🇦 PDF Assistant avec Groq + Qdrant + BGE-M3")

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

# =========================
# EMBEDDING MODEL
# =========================

model = SentenceTransformer("BAAI/bge-m3")

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
    # SPLIT TEXT
    # =========================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    st.info(f"📚 {len(chunks)} chunks créés")

    # =========================
    # EMBEDDINGS
    # =========================

    vectors = model.encode(chunks).tolist()

    # =========================
    # STORE IN QDRANT
    # =========================

    points = []

    for i, chunk in enumerate(chunks):

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
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

        # Embedding question
        question_vector = model.encode(question).tolist()

        # Search
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_vector,
            limit=3
        )

        context = "\n".join(
            [hit.payload["text"] for hit in search_result]
        )

        prompt = f"""
        Réponds à la question en utilisant uniquement le contexte suivant.

        Contexte :
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
