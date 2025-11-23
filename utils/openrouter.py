import requests
from dotenv import load_dotenv
import os

load_dotenv()


from openai import OpenAI
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),  # Replace with your actual API key,
)
completion = client.chat.completions.create(
  model="x-ai/grok-4.1-fast:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ],
  logprobs=True,
)
print(completion.choices[0].logprobs)

