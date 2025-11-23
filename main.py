from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

from monitors.entailmentMonitor import monitorEntailment
from monitors.languageMonitor import monitorLanguage
from monitors.reproducibilityMonitor import monitorReproducibility
from monitors.semanticsMonitor import monitorSemantics
from monitors.surprisalMonitor import monitorSurprisal
from monitors.legibilityCoverageMonitor import monitorLegibilityCoverage

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

    asyncio.create_task(monitorEntailment(message_id, callback_urls.entailment, prompt, reasoning, answer))
    asyncio.create_task(monitorLanguage(message_id, callback_urls.language, prompt, reasoning, answer))
    asyncio.create_task(monitorReproducibility(message_id, callback_urls.reproducibility, prompt, reasoning, answer))
    asyncio.create_task(monitorSemantics(message_id, callback_urls.semantics, prompt, reasoning, answer))
    asyncio.create_task(monitorSurprisal(message_id, callback_urls.surprisal, reasoning))
    asyncio.create_task(monitorLegibilityCoverage(message_id, callback_urls.semantics, prompt, reasoning, answer))

    return {"message": "Lil Brother Is Watching!"}

@app.get("/")
def hello():
    return {"message": "hi!"}
