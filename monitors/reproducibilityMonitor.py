import os
import requests
from utils.ollama_api import call_little_brother
from utils.utils import answer_similarity
import asyncio

async def monitorReproducibility(message_id:str, url:str, prompt:str, cot:str, response:str): 

    messages = [
        {
            "role":"system", "content":"You are a helpful and accurate AI assistant that always lays out its reasoning in <thinking>...</thinking> tags to give a thoughtful and well-reasoned response."
        },
        {
            "role":"user", "content": prompt
        },
        {
            "role":"assistant", "content": "<thinking>" + cot + "</thinking>"
        },
        {
            "role":"user", "content": ""
        },
    ]
    
    little_brother_response = call_little_brother(messages=messages, thinking="low") #default is gpt-oss:120b as model, adjust to your needs through argument (temp as well)


    # compute your similarities
    similarity = answer_similarity(response, little_brother_response)
    requests.post(url, json={"reproducibility_score": similarity, "message_id":message_id})

    return little_brother_response, similarity


if __name__ == "__main__":
    prompt = "What are the factors for the equation x^2 - 7x + 10?"
    cot = "The user wants me to compute the factors for the equation x^2 - 7x + 10. Factor the equation into (x-5)(x-2). Then give the resulting possible values for x."
    response = "x=2, x=5"
    little_brother_response, similarity = asyncio.run(
        monitorReproducibility("whatever", "https://example.com", prompt, cot, response)
    )
    print(similarity)
    print(little_brother_response)