import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Prompt that asks the LLM for 3 different phrasings, one per line
variants_prompt = ChatPromptTemplate.from_template(
    "Write 3 different ways to ask the question below, one per line, "
    "with no numbering and no extra text.\n\nQuestion: {question}"
)
variants_chain = variants_prompt | llm | StrOutputParser()

def multi_query_retrieve(question: str):
    # 1) Get alternative phrasings (and keep the original too)
    variants = variants_chain.invoke({"question": question}).strip().split("\n")
    queries = [question] + [v.strip() for v in variants if v.strip()]
    print("Queries used:", queries)

    # 2) Search with every phrasing and merge, removing duplicates
    seen, merged = set(), []
    for q in queries:
        for d in retriever.invoke(q):
            if d.page_content not in seen:      # skip chunks we already have
                seen.add(d.page_content)
                merged.append(d)
    return merged

for d in multi_query_retrieve("Can I stop my plan?"):
    print("-", d.metadata["source"], "|", d.page_content[:60])




# Code walkthrough
# variants_chain asks the LLM for 3 rephrasings of the question.
# We search with the original plus the 3 variants, so 4 searches in total.
# The seen set removes duplicate chunks, so each chunk appears once in the merged list.
# Expected output: a Queries used: list such as “How do I cancel my subscription?”, “What is the process to end my plan?”, and then merged chunks, most likely the cancellation policy first.