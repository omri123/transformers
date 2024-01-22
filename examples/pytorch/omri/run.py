from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES, LlamaDistributedAttention, LlamaAttention

# device = "cuda" # the device to load the model onto
device = "cpu" # the device to load the model onto

model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

# LLAMA_ATTENTION_CLASSES['eager'] = LlamaAttention
# model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
LLAMA_ATTENTION_CLASSES['eager'] = LlamaDistributedAttention
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True, return_dict_in_generate=False)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])