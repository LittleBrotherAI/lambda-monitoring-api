import requests
from dotenv import load_dotenv
import os

load_dotenv()


from openai import OpenAI
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
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

