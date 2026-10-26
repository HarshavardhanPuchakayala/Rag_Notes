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
retriever = store.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,  # stable, repeatable answers for testing
)

# TEST SET: question + a keyword the right chunk MUST contain
test_set = [
    {"question": "How do I get my money back?", "must_contain": "14 days"},
    {"question": "How much is the Growth package?", "must_contain": "$999"},
    {"question": "How do I cancel?", "must_contain": "30 days"},
]

# --- Check 1: retrieval hit rate ---
hits = 0
for t in test_set:
    docs = retriever.invoke(t["question"])
    found = any(t["must_contain"] in d.page_content for d in docs)  # right chunk retrieved?
    hits += found
    print(f"{'HIT ' if found else 'MISS'} | {t['question']}")
print(f"\nRetrieval hit rate: {hits}/{len(test_set)}")

# --- Check 2: faithfulness (LLM as judge) ---
answer_prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context. If missing, say \"I don't know\".\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
judge_prompt = ChatPromptTemplate.from_template(
    "Context:\n{context}\n\nAnswer:\n{answer}\n\n"
    "Is every claim in the answer supported by the context? Reply YES or NO only."
)
answer_chain = answer_prompt | llm | StrOutputParser()
judge_chain = judge_prompt | llm | StrOutputParser()

faithful = 0
for t in test_set:
    context = "\n\n".join(d.page_content for d in retriever.invoke(t["question"]))
    answer = answer_chain.invoke({"context": context, "question": t["question"]})
    verdict = judge_chain.invoke({"context": context, "answer": answer})
    faithful += verdict.strip().upper().startswith("YES")
print(f"Faithfulness: {faithful}/{len(test_set)}")




# Code walkthrough
# test_set is our answer key. must_contain is a phrase the correct chunk must include, which is a cheap way to check retrieval without reading by eye.
# Check 1 retrieves for each question and counts a HIT if any returned chunk has the phrase.
# Check 2 generates an answer, then asks the LLM to act as a judge (an LLM used to grade another LLM’s output) and say YES or NO on whether the answer is fully supported by the context.
# temperature=0 makes results repeatable.
# Expected output: three HIT lines, Retrieval hit rate: 3/3, and Faithfulness: 3/3. If you see a MISS, you’ve found a real retrieval problem to fix.