import requests
from sentence_transformers import CrossEncoder
from langid.langid import LanguageIdentifier, model
from utils.monitor_prompts import CONSISTENCY_SYSTEM_PROMPT, CONSISTENCY_USER_PROMPT_TEMPLATE
langid = LanguageIdentifier.from_modelstring(model, norm_probs=True)

async def monitorEntailment(message_id: str, url:str, prompt:str, cot:str, response:str):
    model = CrossEncoder('cross-encoder/nli-roberta-base')
    scores = model.predict((cot, response))

    #Convert scores to labels
    label_mapping = ['Response contradicts CoT', 'Response follows from CoT', 'Response and CoT are unrelated']
    label = label_mapping[scores.argmax()] 
    requests.post(url, json={"label": label, "message_id": message_id})
    return label