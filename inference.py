import logging
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.simplefilter("ignore")

device = 'cpu' 
model = AutoModelForCausalLM.from_pretrained("checkpoints/policy_model_1000").to(device)
tokenizer = AutoTokenizer.from_pretrained("Seungyoun/Qwen2.5-7B-Open-R1-Distill")

text = "How many r in starbucks?"

conv = [
    {'role': 'system', 'content': 'think concisely and accurately then answer the question'},
    {'role': 'user', 'content': text},
]

# Convert conversation into input tensors
prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) + "<think>\n"
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
model.generate(input_ids=input_ids, max_new_tokens=512, streamer=streamer)

print()
