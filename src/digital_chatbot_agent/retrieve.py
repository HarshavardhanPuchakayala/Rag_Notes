from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Same embedding model used during indexing (must match!)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Reload the saved vector store from disk
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Turn the store into a retriever that returns the top 2 chunks
retriever = store.as_retriever(search_kwargs={"k": 2})

# Ask a question: the retriever returns a list of Documents
docs = retriever.invoke("What does the Growth package include?")

for i, d in enumerate(docs, 1):
    print(f"\n#{i} from {d.metadata['source']}")
    print(d.page_content)