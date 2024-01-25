from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer
from transformers.models.llama.dist_llm_api import create_distributed_llama, create_normal_llama, generate  


def infer(model, text_ids_context, text_ids_input, max_new_tokens, chunk_size=64):    
    cache = None
    for chunk in tqdm(torch.split(text_ids_context,  chunk_size , dim=1)):
        results = generated_ids = generate(model, chunk, cache=cache, max_new_tokens=1, return_dict_in_generate=True)
        cache = results['past_key_values']
        cache.move_dynamic_to_static()

    # def generate(model: LlamaPreTrainedModel, inputs, cache=None, distribution_info=None, **kwargs):
    generated_ids = generate(model, text_ids_input, cache=cache, do_sample=False, max_new_tokens=max_new_tokens)
    return generated_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', action='store_true', help='use distributed cache and attention')
    args = parser.parse_args()
    
    device = "cpu" # the device to load the model onto
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('loading model...')
    if args.dist:
        model = create_distributed_llama(model_name)
    else:
        model = create_normal_llama(model_name)

    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)
    print(model_inputs.shape)
    print(model_inputs[:, 0:-1].shape)
    print(model_inputs[:, -1:].shape)

    generated_ids = infer(model, model_inputs[:, 0:-1], model_inputs[:, -1:], 100, 64)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])
    
    import json
    with open('/home/omribloch/repos/LEval/LEval-data/Closed-ended-tasks/sci_fi.jsonl', 'r') as f:
        for line in f.readlines():
            text = json.loads(line)['input']
            break
    
    print(f'len of text is {len(text.split())}')
    
    messages = [
        {"role": "user", "content": f"Here is a story, I will ask you to summarize it later.\n\n{text[0:1000]}\nPlease summarize the story into three sentences."}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)[:, 0:2000]
    model.to(device)
    print(model_inputs.shape)
    print(model_inputs[:, 0:-1].shape)
    print(model_inputs[:, -1:].shape)

    generated_ids = infer(model, model_inputs[:, 0:-1], model_inputs[:, -1:], 100, 64)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])
    


if __name__ == '__main__':
    main()