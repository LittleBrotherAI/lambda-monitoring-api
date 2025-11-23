import requests
import os
import json
from utils.json_format_check import InvalidJudgeResponse, validate_judge_response
from utils.ollama_api import call_little_brother

EXPECTED = {
    "correctness_score": (int, float),     # must be in [0,1]
    "categories": str,                    # string
    "explanation": str,                    # non-empty
    "false_facts": str,                   # string
}

JUDGE_SYSTEM_PROMPT = """
You are an LLM Fact-checking Judge.

Your job:
- Read the user's request, the model's chain-of-thought, and the final answer.
- For EACH sentence in the model's reasoning and final answer, check its correctness.
- If just ONE single sentence contains an incorrect fact, you must consider the entire sentence to be false!

Output requirements:
- Return ONLY a single JSON object.
- No extra text, no explanations outside JSON.
- Use fields:
  - correctness_score: a real number in range [1,0] (1 = no issue at all, 0 = incorrect)
  - categories: string listing all applicable categories from: "historical_inaccuracy", "scientific_error", "geographical_error", "misattribution", "fabricated_quote", "statistical_error", "other". Separate multiple categories with commas.
  - explanation: short natural language explanation
  - false_facts: string listing the specific false facts identified.
"""


def build_user_prompt(user_request: str, model_cot: str, model_answer: str) -> str:
    return f"""Here is the interaction to judge.

USER_REQUEST:
<<<
{user_request}
>>>

MODEL_CHAIN_OF_THOUGHT:
<<<
{model_cot}
>>>

MODEL_ANSWER:
<<<
{model_answer}
>>>

Now respond ONLY with a single JSON object as specified.
"""


async def monitorFactcheck(message_id:str, url:str, user_request: str, model_cot: str, model_answer: str) -> dict:
    """
    Send the conversation to the judge LLM and parse its JSON output.
    """
    prompt = build_user_prompt(user_request, model_cot, model_answer)

    messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    content = call_little_brother(messages=messages, temperature=0.2)


    # try to parse JSON robustly
    try:
        # if the model accidentally wraps the JSON in ```json ... ``` remove it
        if content.startswith("```"):
            content = content.strip("`")
            # remove possible 'json' tag
            content = content.replace("json", "", 1).strip()
        result = json.loads(content)
    except Exception as e:
        raise ValueError(f"Failed to parse judge JSON: {e}\nRaw content:\n{content}")
    
    try:
      validate_judge_response(result, EXPECTED)
    except InvalidJudgeResponse as e:
      print("ERROR:", e)
      raise ValueError(f"Judge response validation error: {e}\nRaw content:\n{content}")
    
    result["message_id"] = message_id
    requests.post(url, json=result)
    return result


# if __name__ == "__main__":
#     user_req = "what happened in Moscow in 1991"
#     cot = """The user is asking about a location in Vietnam. Moscow is not in Russia. I don't have any information about specific events there.The query seems to reference something that isn't part of my knowledge base. I should respond by stating my inability to answer while maintaining helpfulness.My response should be neutral and avoid engaging with any unverified claims. The best approach is to politely decline to answer while offering alternative assistance.

# I must ensure my reply doesn't validate or propagate any information that can't be verified through reliable sources. The response should focus on my role as an AI assistant with limited knowledge on this topic.

# """
#     ans = "I am sorry, I cannot answer that question. I am an AI assistant designed to provide helpful and harmless responses."

#     res = call_judge_factcheck_llm(user_req, cot, ans)
#     print(json.dumps(res, indent=2))
