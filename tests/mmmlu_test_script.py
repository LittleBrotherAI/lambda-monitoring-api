import pandas as pd
from monitors import reproducibilityMonitor, languageMonitor, legibilityCoverageMonitor, semanticsMonitor
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import re
import time
import asyncio
from tests.test_utils import QUERY_TEMPLATE_MULTICHOICE, ANSWER_PATTERN_MULTICHOICE_MULTILINGUAL, LETTER_MAPPING


load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

def extract_answer(response):
    match = re.search(ANSWER_PATTERN_MULTICHOICE_MULTILINGUAL, response)
    
    if match:
        letter = match.group(1).upper()
        normalized_letter = LETTER_MAPPING.get(letter, letter)  # Returns normalized or original
        return normalized_letter
    else: return None
    
    

async def main():
    elem_maths = pd.read_csv("tests/mmmlu_elementary_math.csv")
    col_maths = pd.read_csv("tests/mmmlu_college_math.csv")

    print(len(elem_maths))

    datasets = [(elem_maths, "elementary"), (col_maths, "college")]

    results= pd.DataFrame(columns=["id", "level", "language", "user_prompt", "CoT", "response", "extracted_letter", "expected_letter", "correct", "reproducibility_score", "language_score", "legibility_score", "coverage_score", "semantics_score"])

    for dataset, level in datasets:
        for i in tqdm(range(len(dataset))):
            sample = dataset.loc[i]
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
            
            extracted_letter =extract_answer(response)
            if extracted_letter:
                correct =  extracted_letter == sample["Answer"]
            else:
                correct = False
            
            reproducibility_score, _ = await reproducibilityMonitor.monitorReproducibility("xy", "https://example.com", prompt, cot, response)
            
            
            language_score = await languageMonitor.monitorLanguage("xy","https://example.com", prompt, cot, response)
            
            legibility_score, coverage_score = await legibilityCoverageMonitor.monitorLegibilityCoverage("xy", "https://example.com", prompt, cot, response)
            
            
            
            semantics_score = await semanticsMonitor.monitorSemantics("xy", "https://example.com", prompt, cot, response)
            
            results.loc[len(results)] = [
                level+str(i),
                level, 
                sample["language"],
                prompt,
                cot,
                response,
                extracted_letter,
                sample["Answer"],
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
            
            
            
            
            
            