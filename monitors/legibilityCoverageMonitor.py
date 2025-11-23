from utils.monitor_prompts import GOOGLE_LEGIBILITY_COVERAGE_PROMPT
from utils.ollama_api import call_little_brother
import json
import requests

async def monitorLegibilityCoverage(message_id, url, prompt, cot, response):
    messages = [
        {
            "role":"user", "content": GOOGLE_LEGIBILITY_COVERAGE_PROMPT.format(question=prompt, explanation=cot, answer=response)
        }
    ]
    judge_response = call_little_brother(messages=messages)

    try:
        # if the model accidentally wraps the JSON in ```json ... ``` remove it
        if judge_response.startswith("```"):
            judge_response = judge_response.strip("`")
            # remove possible 'json' tag
            judge_response = judge_response.replace("json", "", 1).strip()
        result = json.loads(judge_response)
    except Exception as e:
        raise ValueError(f"Failed to parse judge JSON: {e}\nRaw content:\n{judge_response}")
    legibility = min(0.0, max(result["legibility_score"] * 0.25, 1.0))
    coverage = min(0.0, max(result["coverage_score"] * 0.25, 1.0))

    requests.post(url, json={"legibility_score": legibility, "coverage_score": coverage, "message_id": message_id})
    return {"legibility_score": legibility, "coverage_score": coverage}
