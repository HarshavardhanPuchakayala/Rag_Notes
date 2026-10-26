import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()
# --- Pieces we already built ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})  # the librarian

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

def ask(question: str)->str:
    docs=retriever.invoke(question)
    context="\n\n".join(d.page_content for d in docs)
    filled=prompt.invoke({"context":context, "question":question})
    return llm.invoke(filled).content

print(ask("What does the Growth package include?"))
print(ask("Can I get my money back?"))
print(ask("Do you offer web design?"))  # not in our documents


# ode walkthrough
# We reload the store and retriever exactly like Concept 11, and reuse the template from Concept 12.
# ask() has three lines, one per step: retrieve, fill the prompt, call the LLM.
# "\n\n".join(...) turns the list of Documents into one string, because {context} expects text, not a list.
# Expected output:
# The Growth package: $999/month with SEO, ads management and 5 posts per week.
# Refunds available within 14 days of purchase for monthly packages (ad spend is never refundable).
# Something like “I don’t know”, because web design isn’t in our documents. This proves the “ONLY the context” rule works.
# Common mistakes
# Passing the list of Documents straight into {context} instead of joining their text. The prompt would show messy Document(...) objects.
# Forgetting that llm.invoke(...) returns a message, so you need .content to get the plain text.
# Testing only questions that are in your documents. Always test one that isn’t, to confirm the model says “I don’t know”.