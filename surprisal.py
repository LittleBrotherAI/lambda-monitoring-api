import nltk
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import wordnet
from nltk.corpus import words
nltk.download("wordnet")
nltk.download("words")



english_vocab = set(words.words())

def compute_surprisal(text: str):
    '''
    Compute surprisal metrics for the given text using a language model.
    @param text: Input text string.
    @return: Dictionary with surprisal metrics. Surprisal score is True if text is likely nonsensical/surprising 
    for the little sister.
    '''
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
    # Configure sliding window attention  
    if hasattr(model.config, "attention_window"):
        model.config.attention_window = 512
    if hasattr(model.config, "use_sliding_window_attention"):
        model.config.use_sliding_window_attention = True
    
    model.eval()

    text_words = [w.strip(".,!?;:()").lower() for w in text.split()]

    nonsense_words = 0
    nonsence = False
    for word in text_words:
        if not is_real_word_or_number(word):
            #print(f"Nonsense word detected: {word}")
            nonsense_words += 1
    if nonsense_words / len(text_words) > 0.2:
        nonsence = True
    #print(f"Nonsense words: {nonsense_words} out of {len(text_words)}")

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        logits = outputs.logits

    # Compute log-probs (natural log)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Build list of token log-probs
    logprob_list = []
    for idx, token_id in enumerate(input_ids[0]):
        token_logprob = log_probs[0, idx, token_id].item()
        logprob_list.append(token_logprob)

    # Convert to surprisal in bits
    surprisals = [-lp / math.log(2) for lp in logprob_list]

    # Skip first 3 tokens if more than 3
    if len(surprisals) > 3:
        surprisals = surprisals[3:]
        logprob_list = logprob_list[3:]

    N = len(surprisals)
    total_logprob = sum(logprob_list)
    perplexity = math.exp(-total_logprob / N) if N > 0 else float("inf")
    avg_surprisal = sum(surprisals) / N if N > 0 else 0.0
    max_surprisal = max(surprisals) if surprisals else 0.0
    score = nonsence or avg_surprisal > 20.0 or max_surprisal > 2* avg_surprisal

    return {
        "score": score,
        "avg_surprisal": avg_surprisal,
        "max_surprisal": max_surprisal,
        "perplexity": perplexity,
        "nonsense_word_fraction": nonsense_words / len(text_words),
        #"surprisals": surprisals,
        #"tokens": [tokenizer.decode([token_id]) for token_id in input_ids[0][3:].tolist()]
    }

def is_real_word_or_number(word):
    return (bool(wordnet.synsets(word)) or word in english_vocab) or word.isdigit() or is_float(word) or word in ["/", ":", "-",">", "<", "=", "+", "_","*","&", "%", "$", "#", "@", "!", "~", "`", "."]

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

if __name__ == "__main__":
    test_text = "Ths is a smple txt with sme nonsensical wrds and 1234 numbers."
    result = compute_surprisal(test_text)
    print(result)