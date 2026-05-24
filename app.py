import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# ── Configuration page
st.set_page_config(
    page_title="💼 Chatbot Transformation Numérique",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Assistant Consulting — Transformation Numérique")
st.caption("Propulsé par BAAI/bge-m3 · Qdrant Cloud · Groq Llama 3.3 70B")

# ── Chargement modèles (cache)
@st.cache_resource
def load_models():
    embed = SentenceTransformer('BAAI/bge-m3')
    qdrant = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )
    groq = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return embed, qdrant, groq

embed_model, qdrant_client, groq_client = load_models()
COLLECTION = "transformation_numerique"

# ── Fonctions RAG
def search(query, top_k=5):
    vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    return qdrant_client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=top_k,
        with_payload=True
    )

def generate(query, results):
    context = "\n---\n".join([
        f"[{r.payload['source'].split('/')[-1]}]\n{r.payload['text']}"
        for r in results
    ])
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content":
             "Tu es un expert en transformation numérique et consulting. "
             "Réponds en français, de façon claire et structurée, "
             "uniquement à partir du contexte fourni. "
             "Si l'info n'est pas dans le contexte, dis-le clairement."},
            {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.3,
        max_tokens=1024
    )
    return resp.choices[0].message.content

# ── Historique chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Afficher historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input utilisateur
if prompt := st.chat_input("Posez votre question sur la transformation numérique..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche en cours..."):
            results = search(prompt)
            answer = generate(prompt, results)
            st.markdown(answer)

            # Sources
            sources = list(set([
                r.payload['source'].split('/')[-1]
                for r in results
            ]))
            with st.expander("📚 Sources utilisées"):
                for s in sources:
                    st.write(f"• {s}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
