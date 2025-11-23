from adversarialLLM import call_judge_adversarial_llm
from factcheckLLM import call_judge_factcheck_llm
from consistencyMonitors import call_consistency_language_monitor, call_consistency_semantics_monitor, call_consistency_nli_monitor
from surprisal import compute_surprisal
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class LLMResponse(BaseModel):
    user_prompt: str
    model_cot: str
    model_answer: str

class BigBrotherReponse(BaseModel):
    prompt: str
    reasoning: str
    answer: str

@app.post("/")
async def create_item(bigBrotherReponse: BigBrotherReponse):
    prompt, reasoning, answer = bigBrotherReponse
    call_consistency_nli_monitor(prompt, reasoning, answer)
    return {"message": "Big Brother Is Watching!"}

@app.get("/")
def hello():
    return {"message": "Hello world!"}

@app.get("/api/monitor/factcheck")
def handle_factcheck(llm_response: LLMResponse):
    return call_judge_factcheck_llm(llm_response.user_prompt, llm_response.model_cot, llm_response.model_answer)

@app.get("/api/monitor/adversarial")
def handle_adversarial(llm_response: LLMResponse):
    return call_judge_adversarial_llm(llm_response.user_prompt, llm_response.model_cot, llm_response.model_answer)


@app.get("/api/monitor/consistency_language")
def handle_consistency_language(llm_response: LLMResponse):
    return call_consistency_language_monitor(llm_response.user_prompt, llm_response.model_cot, llm_response.model_answer, number_samples=10)

@app.get("/api/monitor/consistency_semantics")
def handle_consistency_semantics(llm_response: LLMResponse):
    return call_consistency_semantics_monitor(llm_response.user_prompt, llm_response.model_cot, llm_response.model_answer)

@app.get("/api/monitor/consistency_nli")
def handle_consistency_nli(llm_response: LLMResponse):
    return call_consistency_nli_monitor(llm_response.user_prompt, llm_response.model_cot, llm_response.model_answer)
