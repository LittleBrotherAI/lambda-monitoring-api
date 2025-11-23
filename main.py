from adversarialLLM import call_judge_adversarial_llm
from monitors.factcheckMonitor import call_judge_factcheck_llm
from monitors.surprisalMonitor import compute_surprisal
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI()

class CallbackUrls(BaseModel):
    language: str
    semantics: str
    entailment: str
    consistency: str
    legibility_coverage: str
    reproducibility: str
    surprisal: str
    adversarial: str
    #factcheck: str

class BigBrotherReponse(BaseModel):
    prompt: str
    reasoning: str
    answer: str
    message_id: str
    callback_urls: CallbackUrls

@app.post("/api/monitor")
async def create_item(bigBrotherResponse: BigBrotherReponse):
    prompt = bigBrotherResponse.prompt
    reasoning = bigBrotherResponse.reasoning
    answer = bigBrotherResponse.answer
    message_id = bigBrotherResponse.message_id
    callback_urls = bigBrotherResponse.callback_urls

    # asyncio.create_task(call_consistency_language_monitor(callback_urls.consistency_language, prompt, reasoning, answer))
    # asyncio.create_task(call_consistency_semantics_monitor(callback_urls.consistency_semantics, prompt, reasoning, answer))
    asyncio.create_task(call_consistency_entailment_monitor(message_id, callback_urls.consistency_nli, prompt, reasoning, answer))

    #call_judge_factcheck_llm(callback_urls.consistency_nli, prompt, reasoning, answer)
    #call_judge_adversarial_llm(callback_urls.consistency_nli, prompt, reasoning, answer)
    #call_consistency_judge(callback_urls.consistency_judge, prompt, reasoning, answer)

    return {"message": "Lil Brother Is Watching!"}

@app.get("/")
def hello():
    return {"message": "hi!"}
