import pandas as pd
import monitors
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

elem_maths = pd.read_csv("mmmlu_elementary_math.csv")
col_maths = pd.read_csv("mmmlu_college_math.csv")



datasets = [(elem_maths, "elementary"), (col_maths, "college")]

results= pd.DataFrame(columns=["level", "language", "user_prompt", "CoT", "response", "correct"])

for dataset, level in datasets:
    for sample in tqdm(dataset):
        
