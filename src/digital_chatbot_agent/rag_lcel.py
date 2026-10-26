import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- Same pieces as Concept 13 ---
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

# Helper: join retrieved Documents into one text block
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# --- The chain, built with pipes ---
chain = (
    {
        "context": retriever | format_docs,   # question -> retriever -> joined text
        "question": RunnablePassthrough(),    # question passes through unchanged
    }
    | prompt                # fill the template
    | llm                   # LLM writes the answer
    | StrOutputParser()     # message -> plain string
)

# One call runs the whole pipeline
print(chain.invoke("What does the Growth package include?"))
print(chain.invoke("Do you offer web design?"))