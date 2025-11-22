import os
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

def call_little_brother(messages, model="gpt-oss:120b-cloud", thinking="high", return_thinking = False, temperature:float=0.2):
    client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    )


    response = client.chat(model=model, messages=messages, think=thinking, options={"temperature":temperature})

    if return_thinking:
        return "<thinking>" + response.message.thinking + "</thinking>\n\n" + response.message.content
    else:
        return response.message.content

  