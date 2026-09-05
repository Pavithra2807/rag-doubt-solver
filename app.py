import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="AI/ML Doubt Solver", page_icon="🤖")

# Cache the heavy stuff so it only loads once, not on every question
@st.cache_resource
def load_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return vectordb, tokenizer, model

vectordb, tokenizer, model = load_pipeline()

st.title("🤖 AI/ML Doubt Solver")
st.caption("A RAG-based chatbot that answers AI/ML questions using retrieved notes as context.")

question = st.text_input("Ask your AI/ML doubt:")

if question:
    with st.spinner("Retrieving relevant notes and generating answer..."):
        results = vectordb.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in results])

        prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {question}
Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("See retrieved context"):
        st.write(context)