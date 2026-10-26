from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

question = "Do you offer web design?"  # try a question that gave a bad answer

# 1) How many chunks are in the store? (0 means indexing failed)
print("Chunks in store:", store._collection.count())

# 2) What did the retriever return, and how strong were the matches?
results = store.similarity_search_with_score(question, k=3)  # returns (Document, distance)
for doc, score in results:
    print(f"\nscore={score:.3f} | source={doc.metadata.get('source')}")
    print(doc.page_content)  # read the FULL chunk, not a short preview