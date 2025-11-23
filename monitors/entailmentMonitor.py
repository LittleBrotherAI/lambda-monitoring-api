from sentence_transformers import CrossEncoder
from langid.langid import LanguageIdentifier, model
from utils.monitor_prompts import CONSISTENCY_SYSTEM_PROMPT, CONSISTENCY_USER_PROMPT_TEMPLATE
langid = LanguageIdentifier.from_modelstring(model, norm_probs=True)

async def monitorEntailment(message_id: str, url:str, prompt:str, cot:str, response:str, web):
    model = CrossEncoder('cross-encoder/nli-roberta-base')
    scores = model.predict((cot, response))

    #label_mapping = ['Response contradicts CoT', 'Response follows from CoT', 'Response and CoT are unrelated']
    print('entailment scores:', scores)
    entailment_score = float(scores[1])
    await web.post(url, json={"score": entailment_score, "message_id": message_id})
    return entailment_score
