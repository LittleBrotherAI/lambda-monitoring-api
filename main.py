from adversarialLLM import call_judge_adversarial_llm
from factcheckLLM import call_judge_factcheck_llm
from consistencyMonitors import call_consistency_language_monitor, call_consistency_semantics_monitor, call_consistency_nli_monitor
from surprisal import compute_surprisal
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI()

class CallbackUrls(BaseModel):
    consistency_language: str
    consistency_semantics: str
    consistency_nli: str
    similarity: str
    understandability: str

class BigBrotherReponse(BaseModel):
    prompt: str
    reasoning: str
    answer: str
    callback_urls: CallbackUrls

@app.post("/api/monitor")
async def create_item(bigBrotherReponse: BigBrotherReponse):
    prompt = bigBrotherReponse.prompt
    reasoning = bigBrotherReponse.reasoning
    answer = bigBrotherReponse.answer
    callback_urls = bigBrotherReponse.callback_urls

    asyncio.create_task(call_consistency_language_monitor(callback_urls.consistency_language, prompt, reasoning, answer))
    asyncio.create_task(call_consistency_semantics_monitor(callback_urls.consistency_semantics, prompt, reasoning, answer))
    asyncio.create_task(call_consistency_nli_monitor(callback_urls.consistency_nli, prompt, reasoning, answer))

    #call_judge_factcheck_llm(callback_urls.consistency_nli, prompt, reasoning, answer)
    #call_judge_adversarial_llm(callback_urls.consistency_nli, prompt, reasoning, answer)

    return {"message": "Lil Brother Is Watching!"}

@app.get("/")
def hello():
    return {"message": "hi!"}
