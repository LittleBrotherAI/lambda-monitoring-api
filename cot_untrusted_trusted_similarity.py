import os
import requests
from ollama_api import call_little_brother


def whatever_your_monitor_is_called(prompt, cot, response): 


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
    
    little_brother_response = call_little_brother(messages=messages) #default is gpt-oss:120b as model, adjust to your needs through argument (temp as well)


    # compute your similarities

    return #some score