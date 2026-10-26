from langchain_core.prompts import ChatPromptTemplate

# The template: fixed rules + two blanks ({context} and {question})
prompt = ChatPromptTemplate.from_template(
    """You are BrightPath Digital's helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}"""
)

# Pretend these came from the retriever
fake_context = "Refunds are available within 14 days of purchase. Ad spend is never refundable."

# Fill the blanks to get the final prompt
filled = prompt.invoke({
    "context": fake_context,          # goes into {context}
    "question": "Can I get a refund?" # goes into {question}
})

print(filled.to_string())  # see exactly what the LLM will receive