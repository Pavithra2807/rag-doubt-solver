from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load the same embeddings model used during ingestion
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load the existing vector database (created by ingest.py)
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# 3. Load the local LLM directly (no pipeline() wrapper)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# 4. Take a question from the user
question = input("Ask your AI/ML doubt: ")

# 5. Retrieve the most relevant chunks
results = vectordb.similarity_search(question, k=3)

# 6. Combine retrieved chunks into context
context = "\n\n".join([doc.page_content for doc in results])

# 7. Build a prompt with context + question
prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {question}
Answer:"""

# 8. Generate the answer directly using the model
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
outputs = model.generate(**inputs, max_new_tokens=150)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- Retrieved Context ---")
print(context)
print("\n--- Answer ---")
print(answer)