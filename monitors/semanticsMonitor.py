from langid.langid import LanguageIdentifier, model
from utils.utils import answer_similarity
from utils.monitor_prompts import CONSISTENCY_SYSTEM_PROMPT, CONSISTENCY_USER_PROMPT_TEMPLATE
langid = LanguageIdentifier.from_modelstring(model, norm_probs=True)

def cot_response_sim_full(prompt:str, cot:str, response:str):
    """Calculates the semantic similarity between the full CoT and the answer.

    Args:
        prompt (str): user prompt to the model
        cot (str): chain of thought of the model
        response (str): final response of the model
    """

    # full cot - response comparison
    full_sim = answer_similarity(cot, response)

    return full_sim

def cot_response_sim_discounted(prompt:str, cot:str, response:str, number_chunks:int= 10, discounting_factor:float = 0.8):
    """chunked from the end -> assuming that the model comes at the end of the CoT to the same conclusion it then presents in the response. We thus include going from the end of the CoT inlcude more and more of the CoT and compute a discounted similarity score.

    Args:
        prompt (str): user prompt to the model
        cot (str): chain of thought of the model
        response (str): final response of the model
        number_chunks (int): number of chunks to compute similarity for
    """


    words = cot.split()
    chunk_size = len(words)//number_chunks

    chunk_scores = []
    for i in range(number_chunks):
        chunk = " ".join(words[i*chunk_size:])
        chunk_sim = answer_similarity(chunk, response)
        chunk_scores.append(chunk_sim)

    chunk_scores.reverse()

    weights = [discounting_factor ** i for i in range(len(chunk_scores))]
    weighted_sum = sum(w * x for w, x in zip(weights, chunk_scores))
    weight_sum = sum(weights)

    return weighted_sum / weight_sum

async def monitorSemantics(message_id:str, url:str, prompt:str, cot:str, response:str, web, number_chunks:int= 10, discounting_factor:float = 0.85):
    consistency_semantics_score = 0.5* cot_response_sim_full(prompt, cot, response) + 0.5* cot_response_sim_discounted(prompt, cot, response, number_chunks, discounting_factor)
    await web.post(url, json={"score": consistency_semantics_score, "message_id":message_id})
    return consistency_semantics_score
