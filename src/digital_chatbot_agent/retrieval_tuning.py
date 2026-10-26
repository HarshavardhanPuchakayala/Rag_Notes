from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

question = "Tell me about refunds and cancellation"

# 1) MMR: relevant AND diverse
mmr_retriever = store.as_retriever(
    search_type="mmr",                                   # use MMR instead of plain similarity
    search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.5},
    # fetch_k: how many candidates to consider first
    # lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity, 0.5 = balanced
)
print("MMR results:")
for d in mmr_retriever.invoke(question):
    print("-", d.metadata["source"], "|", d.page_content[:60])

# 2) Metadata filter: search ONLY inside policies.txt
policy_retriever = store.as_retriever(
    search_kwargs={"k": 2, "filter": {"source": "policies.txt"}}  # Chroma filter syntax
)
print("\nFiltered (policies only):")
for d in policy_retriever.invoke("What does Growth include?"):
    print("-", d.metadata["source"], "|", d.page_content[:60])









# Code walkthrough
# search_type="mmr" switches the ranking. It first fetches fetch_k candidates, then picks k that balance relevance and variety.
# lambda_mult is the dial between “most relevant” and “most different”.
# "filter": {"source": "policies.txt"} makes Chroma search only chunks whose metadata matches.
# Expected output: the MMR list shows 2 chunks that cover different ideas (with our tiny data, likely refund and cancellation). The filtered list shows only policies.txt chunks, even though the question asks about Growth, which lives in packages.txt. This proves the filter really restricts the search, and also shows its risk.