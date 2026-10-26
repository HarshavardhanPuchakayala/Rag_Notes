import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_DIR="./chroma_db"  # where the vector store is saved


# --- Sample files so the script runs on its own ---
with open("packages.txt", "w", encoding="utf-8") as f:
    f.write("BrightPath Digital packages:\n"
            "Starter: $499/month, SEO and 2 social posts per week.\n"
            "Growth: $999/month, SEO, ads management and 5 posts per week.")
with open("policies.txt", "w", encoding="utf-8") as f:
    f.write("Refund Policy: Refunds are available within 14 days of purchase "
            "for monthly packages. Ad spend is never refundable.\n\n"
            "Cancellation Policy: You can cancel any package with 30 days "
            "written notice.")


# 1) LOAD: read every file into Documents
docs = []
for path in ["packages.txt", "policies.txt"]:
    docs.extend(TextLoader(path, encoding="utf-8").load())  # add each file's Documents
print("Loaded documents:", len(docs))

#2) SPLIT: cut Documents into overlapping chunks

splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks=splitter.split_documents(docs)
print("chunks created:",len(chunks))

# 3) EMBED + STORE: turn chunks into vectors and save them

if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR) # start fresh so we never store duplicates

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
Chroma.from_documents(chunks,embeddings ,persist_directory=DB_DIR)
print("Saved to:", DB_DIR)

store=Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
for r in store.similarity_search("How do i get my money back?", k=1):
    print("TOP MATCH:", r.page_content[:80], "|",r.metadata)
