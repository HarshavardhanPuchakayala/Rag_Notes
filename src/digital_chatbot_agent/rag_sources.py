import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

# --- Same pieces as Concept 14 ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    """You are BrightPath Digital's helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}"""
)

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Part 1: given {"docs":..., "question":...}, produce the answer string
answer_chain = (
    {
        "context": lambda x: format_docs(x["docs"]),  # join the retrieved chunks
        "question": lambda x: x["question"],          # pick out the question
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Part 2: retrieve once, keep the docs, then add the answer next to them
chain = (
    RunnableParallel(docs=retriever, question=RunnablePassthrough())  # fetch docs, keep question
    .assign(answer=answer_chain)                                      # add the "answer" key
)

result = chain.invoke("Can I get my money back?")

print("ANSWER:", result["answer"])
print("\nSOURCES:")
for d in result["docs"]:
    print("-", d.metadata["source"], "|", d.page_content[:60], "...")


#     Code walkthrough
# RunnableParallel(docs=retriever, question=RunnablePassthrough()) runs the retriever once and keeps both the Documents and the original question in a dictionary.
# .assign(answer=answer_chain) adds a new key, answer, by running the rest of the chain on that dictionary. The docs and question keys stay in the result.
# The two lambda functions pull docs and question out of the dictionary for the prompt.
# result is now a dictionary: result["answer"] is the text, and result["docs"] is the list of Documents with their metadata.
# Expected output: an ANSWER: line about the 14-day refund window, then a SOURCES: list with entries like - policies.txt | Refund Policy: Refunds are available within 14 days .... Because k=3 and our knowledge base is tiny, you may also see packages.txt listed, which shows the retriever returns the top 3 even when some are less relevant.
# Common mistakes
# Retrieving twice (once for the answer, once for the sources). The two sets could differ, and you’d show sources that didn’t produce the answer. Retrieve once and reuse.
# Citing every retrieved chunk as if it was used. The list shows what the LLM received, not necessarily what it relied on.
# Losing metadata during loading or splitting, which leaves you with empty sources. Page numbers from PDFs are very useful here (d.metadata["page"]).