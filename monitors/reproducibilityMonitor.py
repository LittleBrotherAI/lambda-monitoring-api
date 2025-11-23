import os
import requests
from utils.ollama_api import call_little_brother
from utils.utils import answer_similarity

async def monitorReproducibility(message_id:str, url:str, prompt:str, cot:str, response:str)->float: 

    messages = [
        {
            "role":"system", "content":"You are a helpful and accurate AI assistant that always lays out its reasoning in <thinking>...</thinking> tags to give a thoughtful and well-reasoned response."
        },
        {
            "role":"user", "content": prompt
        },
        {
            "role":"assistant", "content": "<thinking>" + cot + "</thinking>"
        }
    ]
    
    little_brother_response = call_little_brother(messages=messages, thinking="low") #default is gpt-oss:120b as model, adjust to your needs through argument (temp as well)


    # compute your similarities
    similarity = answer_similarity(response, little_brother_response)
    requests.post(url, json={"reproducibility_score": similarity, "message_id":message_id})

    return similarity