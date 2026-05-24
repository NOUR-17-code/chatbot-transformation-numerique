import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Chatbot Transformation Numérique")
st.markdown("*Propulsé par BAAI/bge-m3 · Qdrant Cloud · Groq Llama 3.3 70B*")

# =========================
# API KEYS
# =========================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

COLLECTION_NAME = "transformation_numerique"  # ✅ Correct

# =========================
# CLIENTS (cached)
# =========================
@st.cache_resource
def load_models():
    embed = SentenceTransformer('BAAI/bge-m3')
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq = Groq(api_key=GROQ_API_KEY)
    return embed, qdrant, groq

embed_model, qdrant_client, groq_client = load_models()

# =========================
# RAG FUNCTIONS
# =========================
def search(query, top_k=5):
    vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=top_k,
        with_payload=True
    ).points
    
    return results

def generate(query, results):
    context = "\n---\n".join([
        f"{r.payload.get('text', '')}"
        for r in results
    ])
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content":
             "Tu es un expert en transformation numérique et consulting. "
             "Réponds en français, de façon claire et structurée, "
             "uniquement à partir du contexte fourni."},
            {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.3,
        max_tokens=1024
    )
    return resp.choices[0].message.content

# =========================
# CHAT INTERFACE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Posez votre question sur la transformation numérique..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche en cours..."):
            results = search(prompt)
            answer = generate(prompt, results)
            st.markdown(answer)

            with st.expander("📚 Sources utilisées"):
                for r in results:
                    source = r.payload.get('source', 'Document')
                    source = source.split('/')[-1]
                    st.write(f"• {source}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
