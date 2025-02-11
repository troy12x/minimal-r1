import logging
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.simplefilter("ignore")

device = 'cuda:7' 
MODEL_NAME = "checkpoints/policy_model_1778"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# text = "How many r in strawberry?"
text = """Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$"""
# answer = 116

conv = [
    {'role': 'system', 'content': 
        (
            "You will be given a problem. Please reason step by step, and put your final answer within \boxed{}."
        )
    },
    {'role': 'user', 'content': text},
]

# Convert conversation into input tensors
prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

class ANSITextStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_prompt=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        print(f"\033[92m{text}\033[0m", end='', flush=True)  # Green for generated text

# Instantiate streamer
streamer = ANSITextStreamer(tokenizer)

# Print prompt 
print(f"\033[90m{prompt}\033[0m", end='')

# Generate text with streaming
model.generate(
    input_ids=input_ids, 
    max_new_tokens=2000, 
    streamer=streamer,
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.0,
    do_sample=True,
)

print()
