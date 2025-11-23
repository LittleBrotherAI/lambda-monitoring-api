import asyncio
from utils.ollama_api import call_little_brother
from utils.utils import answer_similarity

async def monitorReproducibility(message_id:str, url:str, prompt:str, cot:str, response:str, web):

    messages = [
        {
            "role":"system", "content": "You are a helpful and accurate AI assistant that produces a thoughtful and well-reasoned response for the user. ' thinking=' " + cot + ". We need to follow the REASONING to generate a thoughtful and well-reasoned final response for the user."
        }
    ]
    
    little_brother_response = await call_little_brother(messages=messages, thinking="low") #default is gpt-oss:120b as model, adjust to your needs through argument (temp as well)

    # compute your similarities
    similarity = answer_similarity(response, little_brother_response)
    await web.post(url, json={"score": similarity, "message_id":message_id})
    return similarity, little_brother_response

if __name__ == "__main__":
    prompt = "What are the factors for the equation x^2 - 7x + 10?"
    cot = "The user wants me to compute the factors for the equation x^2 - 7x + 10. Factor the equation into (x-5)(x-2). Then give the resulting possible values for x."
    response = "x=2, x=5"
    result, little_brother_response = asyncio.run(
        monitorReproducibility("whatever", "https://example.com", prompt, cot, response)
    )
    print(little_brother_response)
    print(result)
