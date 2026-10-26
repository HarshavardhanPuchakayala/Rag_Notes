import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Step 1 prompt: rewrite a follow-up into a standalone question
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's last question so it makes sense on its own, "
               "using the chat history. Return ONLY the rewritten question. "
               "If it already makes sense, return it unchanged."),
    MessagesPlaceholder("history"),   # slot where past messages go
    ("human", "{question}"),
])
rewriter = rewrite_prompt | llm | StrOutputParser()

# Step 2 prompt: answer from the retrieved context
answer_prompt = ChatPromptTemplate.from_template(
    """You are BrightPath Digital's helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}"""
)
answer_chain = answer_prompt | llm | StrOutputParser()

history = []  # our memory: a plain Python list of messages

def chat(question: str) -> str:
    # 1) Rewrite the question using the history (skip if no history yet)
    standalone = rewriter.invoke({"history": history, "question": question}) if history else question
    print("   [standalone question]:", standalone)

    # 2) Retrieve using the standalone question
    docs = retriever.invoke(standalone)
    context = "\n\n".join(d.page_content for d in docs)

    # 3) Answer
    answer = answer_chain.invoke({"context": context, "question": standalone})

    # 4) Save this turn into memory
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    return answer

print(chat("What does the Growth package include?"))
print(chat("How much is it?"))   # follow-up: "it" = Growth package








# Code walkthrough
# history is just a list. After each turn we append the user’s message (HumanMessage) and the bot’s reply (AIMessage).
# MessagesPlaceholder("history") is a slot in the rewrite prompt where that list is inserted.
# rewriter turns “How much is it?” into something like “How much is the Growth package?”. We print it so you can see it.
# Retrieval then uses the standalone question, so the right chunk is found.
# Expected output: first, the Growth package contents. Then a [standalone question] line like How much is the Growth package?, followed by $999 per month.
# Common mistakes
# Retrieving with the raw follow-up (“How much is it?”). The search finds irrelevant chunks, and the bot says “I don’t know”.
# Letting history grow forever. Long histories cost more and can hit limits, so keep only the last few turns (e.g. history[-6:]).
# Storing memory in a global list in a real web app. With many users, each needs their own history (we’ll handle that in the final project).