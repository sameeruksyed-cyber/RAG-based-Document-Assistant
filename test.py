import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to the .env file.")

llm = ChatGroq(
    groq_api_key=api_key,
    model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
)

response = llm.invoke("Say hello")

print(response.content)