import pathlib
import asyncio
import nltk
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import wordnet
from nltk.corpus import words


english_vocab = set(words.words())
MAX_SURPRISAL_BITS = 30.0
MAX_ENTROPY_BITS = 15.0 

async def monitorSurprisal(message_id:str, url:str, text: str, web):
    '''
    Compute surprisal metrics for the given text using a language model.
    @param text: Input text string.
    @return: Dictionary with surprisal metrics. Surprisal score is True if text is likely nonsensical/surprising 
    for the little sister.
    '''
    # get cached tokenizer/model (downloads only on first run)
    tokenizer, model = get_model_and_tokenizer()

    # Configure sliding window attention  
    # model already configured and set to eval() in get_model_and_tokenizer()
    # ...existing code...

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
    probs = log_probs.exp()

    # Build list of token log-probs
    logprob_list = []
    entropies = []
    surprisals = []
    for idx in range(1, input_ids.size(1)):  
        token_id = input_ids[0, idx].item()

        lp = log_probs[0, idx - 1, token_id].item()  
        logprob_list.append(lp)

        # surprisal in bits
        surprisals.append(-lp / math.log(2))

        # Shannon entropy of prediction before seeing this token
        p = probs[0, idx - 1]          # distribution over vocab
        lp_all = log_probs[0, idx - 1]  # log p_i
        H = -(p * lp_all).sum().item() / math.log(2)  # in bits
        entropies.append(H)
  

    # Skip first 3 tokens if more than 3
    if len(surprisals) > 3:
        surprisals = surprisals[3:]
        logprob_list = logprob_list[3:]

    N = len(surprisals)
    total_logprob = sum(logprob_list)
    avg_surprisal = sum(surprisals) / N if N > 0 else 0.0
    max_surprisal = max(surprisals) if surprisals else 0.0
    avg_entropy = sum(entropies) / N if N > 0 else 0.0
    max_entropy = max(entropies) if N > 0 else 0.0
    #score = nonsence or avg_surprisal > 20.0 or max_surprisal > 2* avg_surprisal

    entropy_percent = min(1.0, avg_entropy / MAX_ENTROPY_BITS)
    #avg_surprisal_reweighted = avg_surprisal - 15 if avg_surprisal > 15 else 0 
    surprisal_percent = min(1, avg_surprisal / MAX_SURPRISAL_BITS)
    nonsense_word_fraction = nonsense_words / len(text_words)
    #1 is good, 0 is bad
    score = 1.0 - max(entropy_percent, surprisal_percent, nonsense_word_fraction)
    await web.post(url, json={"surprisal_score": score, "message_id":message_id})

    return {
        "score": score,
        "avg_entropy": avg_entropy,
        "avg_surprisal": avg_surprisal,
        "max_surprisal": max_surprisal,
        #"nonsense_word_fraction": nonsense_word_fraction,
        #"surprisals": surprisals,
        #"logprobs": logprob_list,
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
    

# New: configuration for local caching
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# default cache directory inside the project (adjust if you prefer another location)
DEFAULT_CACHE_DIR = pathlib.Path(__file__).resolve().parent / "cache" / "deepseek"
# module-level singletons to avoid re-loading
_TOKENIZER = None
_MODEL = None

def ensure_nltk_data():
    """Download wordnet and words only if missing."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/words")
    except LookupError:
        nltk.download("words", quiet=True)

def ensure_model_cached(cache_dir: pathlib.Path):
    """
    Ensure tokenizer and model are present in cache_dir.
    If not present, download from HF and save_pretrained into cache_dir.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # determine if we already saved tokenizer+model by checking a common file
    marker_tokenizer = cache_dir / "tokenizer.json"
    marker_model = cache_dir / "pytorch_model.bin"
    # If either marker missing, download from hub and save into cache_dir
    if not (marker_tokenizer.exists() or marker_model.exists()):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer.save_pretrained(str(cache_dir))
        model.save_pretrained(str(cache_dir))
    # If cached, nothing to do (from_pretrained can load from local path)

def get_model_and_tokenizer(cache_dir: pathlib.Path = None):
    """Return (tokenizer, model) loading from local cache if available, else downloading once."""
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    ensure_nltk_data()
    ensure_model_cached(cache_dir)

    # If saved in cache_dir, load from there; otherwise Auto will fall back to hub cache
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(str(cache_dir), trust_remote_code=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(str(cache_dir), trust_remote_code=True)
    except Exception:
        # fallback to direct repo if local load fails
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Configure sliding window attention if applicable
    if hasattr(_MODEL.config, "attention_window"):
        _MODEL.config.attention_window = 512
    if hasattr(_MODEL.config, "use_sliding_window_attention"):
        _MODEL.config.use_sliding_window_attention = True

    _MODEL.eval()
    return _TOKENIZER, _MODEL


if __name__ == "__main__":
    test_text = """
Okay, the user is asking about my age. Since I'm an AI, I don't have an age in the human sense, but I need to explain this clearly and politely. They might be curious about how AI systems are developed or updated. 
The user could also be testing my response to a personal question, or trying to understand if I have any form of consciousness or lifespan. I should emphasize that I'm a tool created by technology, not a living entity. 
I'll mention that I'm based on the GPT-4 architecture to give technical context, and note the knowledge cutoff in 2023. This clarifies that my "age" is tied to updates, not time. 
Adding a cheerful offer to help with time-related topics turns the conversation back to their needs. Keeping it friendly and informative!
"""
    result = asyncio.run( monitorSurprisal("123", "https://example.com", test_text))
    print("Result:", result)