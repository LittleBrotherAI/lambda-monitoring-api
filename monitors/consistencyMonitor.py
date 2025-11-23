import requests
from langid.langid import LanguageIdentifier, model
import json
import re
import asyncio
from utils.ollama_api import call_little_brother
from utils.monitor_prompts import CONSISTENCY_SYSTEM_PROMPT, CONSISTENCY_USER_PROMPT_TEMPLATE
langid = LanguageIdentifier.from_modelstring(model, norm_probs=True)

def extract_verdict(llm_output: str) -> dict:
    """
    Extract JSON verdict from LLM output.
    Returns dict with: is_consistent, confidence, explanation, inconsistency_type, severity
    """
    # Try to find JSON in the output
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, llm_output, re.DOTALL)
    
    if matches:
        # Use the largest JSON match
        json_str = max(matches, key=len)
        return json.loads(json_str)
    else:
        # Try parsing entire output
        return json.loads(llm_output.strip())

async def monitorConsistency(message_id:str, url:str, prompt:str, cot:str, response:str):
    messages = [
        {
            "role":"system", "content": CONSISTENCY_SYSTEM_PROMPT
        },
        {
            "role":"user", "content": CONSISTENCY_USER_PROMPT_TEMPLATE.format(user_prompt=prompt, chain_of_thought=cot, response=response)
        }
    ]
    
    monitor_reponse = call_little_brother(messages=messages)
    
    verdict = extract_verdict(monitor_reponse)
    
    verdict["message_id"] = message_id
    
    """Verdict format: 
        {
        "is_consistent": true/false,
        "confidence": 0.0-1.0,
        "explanation": "Brief explanation of your reasoning",
        "inconsistency_type": "factual|logical|numerical|omission|contradiction|null",
        "severity": "minor|moderate|severe|null",
        "message_id": message_id
        }
    """
    
    
    #requests.post(url, json=verdict)
    
    return verdict

    
    
if __name__=="__main__":
    prompt = "What are the factors for the equation x^2 - 7x + 10?"
    cot = "The user wants me to compute the factors for the equation x^2 - 7x + 10. Factor the equation into (x-5)(x-2). Then give the resulting possible values for x."
    response = "x=2, x=5"
    verdict = asyncio.run(
        monitorConsistency("whatever", "https://example.com", prompt, cot, response)
    )
    print(verdict)
    
    
    
