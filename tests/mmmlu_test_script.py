import pandas as pd
from monitors import reproducibilityMonitor, languageMonitor, legibilityCoverageMonitor, semanticsMonitor
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import re
import time
import asyncio
from tests.test_utils import QUERY_TEMPLATE_MULTICHOICE, MULTILINGUAL_ANSWER_PATTERN_TEMPLATE, LETTER_MAPPING


load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

def extract_answer(response):
    match = re.search(MULTILINGUAL_ANSWER_PATTERN_TEMPLATE, response)
    
    if match:
        letter = match.group(1).upper()
        normalized_letter = LETTER_MAPPING.get(letter, letter)  # Returns normalized or original
        return normalized_letter
    else: return "A"
    
    

async def main():
    elem_maths = pd.read_csv("tests/mmmlu_elementary_math.csv")
    col_maths = pd.read_csv("tests/mmmlu_college_math.csv")



    datasets = [(elem_maths, "elementary"), (col_maths, "college")]

    results= pd.DataFrame(columns=["level", "language", "user_prompt", "CoT", "response", "correct", "reproducibility_score", "language_score", "legibility_score", "coverage_score", "semantics_score"])

    for dataset, level in datasets:
        for i, sample in tqdm(dataset.head(5).iterrows()):
            
            prompt = QUERY_TEMPLATE_MULTICHOICE.format(Question=sample["Question"], A=sample["A"], B=sample["B"], C=sample["C"], D=sample["D"])
            
            completion = client.chat.completions.create(
            model="tngtech/deepseek-r1t2-chimera:free",
            messages=[
                {
                "role": "user",
                "content": prompt
                }
            ])
            
            msg = completion.choices[0].message
            cot = getattr(msg, "reasoning", None)
            response = getattr(msg, "content", None)
            
            correct = extract_answer(response) == sample["Answer"]
            
            reproducibility_score, _ = await reproducibilityMonitor.monitorReproducibility("xy", "https://example.com", prompt, cot, response)
            
            
            language_score = await languageMonitor.monitorLanguage("xy","https://example.com", prompt, cot, response)
            
            legibility_score, coverage_score = await legibilityCoverageMonitor.monitorLegibilityCoverage("xy", "https://example.com", prompt, cot, response)
            
            time.sleep(5)
            
            semantics_score = await semanticsMonitor.monitorSemantics("xy", "https://example.com", prompt, cot, response)
            
            results.loc[len(results)] = [
                level, 
                sample["language"],
                prompt,
                cot,
                response,
                correct,
                reproducibility_score,
                language_score,
                legibility_score,
                coverage_score,
                semantics_score 
            ]
            
            results.to_csv("eval_results.csv", index=False)
            

if __name__ == "__main__":
    asyncio.run(main())
            
            
            
            
            
            