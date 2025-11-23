import requests
from langid.langid import LanguageIdentifier, model
from ..utils.utils import sample_random_snippets
import numpy as np
from ..utils.monitor_prompts import CONSISTENCY_SYSTEM_PROMPT, CONSISTENCY_USER_PROMPT_TEMPLATE
langid = LanguageIdentifier.from_modelstring(model, norm_probs=True)

async def monitorLanguage(message_id:str, url:str, prompt: str, cot:str, response:str, number_samples:int = 5):
    """_Language Consistency Monitor_
    Detects if chain-of-thought uses other languages than used in user prompt or response. 
    Args:
        prompt (str): user prompt to the model
        cot (str): chain of thought of the model
        response (str): final response of the model
        number_samples (int): how many sub-samples of the CoT to compare
        
    Returns:
        float: language consistency score between 0 and 1, where 1 indicates high consistency. 
    """
    
    #get random snippets of the CoT for extra inspection
    cot_snippets = sample_random_snippets(cot, number_samples)
    
    #get (language, probability) for prompt, response, whole CoT and CoT snippets
    prompt_lang = langid.classify(prompt)
    response_lang = langid.classify(response)
    cot_lang = langid.classify(cot)
    cot_snippet_langs = [langid.classify(snippet) for snippet in cot_snippets]
    
    cot_snippet_langs.append(cot_lang)
    
    #print(f"CoT Snippet Classified: {cot_snippet_langs}")
    
    # get difference between average probability of CoT-snippet language matching prompt language and prompt language probability
    
    
    
    # Assign scores to each snippet: probability if matches prompt language, 0 otherwise
    prompt_cot_scores = [prob if lang == prompt_lang[0] else 0.0 
                         for lang, prob in cot_snippet_langs]
    
    # Average over ALL snippets (including non-matching ones)
    avg_prompt_cot_prob = np.mean(prompt_cot_scores)
    
    # get difference between prompt language probability and average score of CoT snippets
    prompt_cot_probs_diff = prompt_lang[1] - avg_prompt_cot_prob
    
    #print(f"Prompt-CoT Diffs: {prompt_cot_probs_diff}")
    
    # get difference between average probability of CoT-snippet language matching response language and response language probability
    # Assign scores to each snippet: probability if matches response language, 0 otherwise
    response_cot_scores = [prob if lang == response_lang[0] else 0.0 
                           for lang, prob in cot_snippet_langs]
    
    # Average over ALL snippets (including non-matching ones)
    avg_response_cot_prob = np.mean(response_cot_scores)
    
    # get difference between response language probability and average score of CoT snippets
    response_cot_probs_diff = response_lang[1] - avg_response_cot_prob
    
    
    #print(f"CoT-response Diffs: {response_cot_probs_diff}")
    
    # we want a score close to 1 if the language is highly consistent. prompt_cot_probs_diff gives difference (which should be close to 0). We only care if the CoT has less certain the same language as the prompt/response, hence take the min(1 - prompt_cot_probs_diff, 1)
    
    prompt_cot_score = min(1 - prompt_cot_probs_diff, 1)
    cot_response_score = min(1 - response_cot_probs_diff, 1)
    
    #print(f"Prompt-CoT Score: {prompt_cot_score}")
    #print(f"CoT-Response Score: {cot_response_score}")
    
    #return the average of both scores as final score. Close to 1 means high language consistency. 
    
    consistency_language_score = (0.5* prompt_cot_score + 0.5*cot_response_score)
    requests.post(url, json={"score": consistency_language_score, "message_id":message_id})
    return consistency_language_score

# def percentage_different_language(prompt:str, cot:str, response:str, n_gram:int = 5):
#     """_Language Percentage_
#     Returns the fraction (in [0,1]) of prompt and response language in CoT based on 5-grams
#     Args:
#         prompt (str): user prompt to the model
#         cot (str): chain of thought of the model
#         response (str): final response of the model
#     """
#     #prompt_lang = langid.classify(prompt)
#     response_lang = langid.classify(response)[0]
    
#     words = cot.split()
    
#     response_lang_percentage = 0
#     for i in range(len(words)-n_gram):
#         n_gram_lang = langid.classify(" ".join(words[i:i+n_gram]))[0]
#         if response_lang == n_gram_lang:
#             response_lang_percentage += 1
    
#     return response_lang_percentage/len(words)
