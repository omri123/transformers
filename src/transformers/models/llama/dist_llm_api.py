import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DistributedCache, DynamicCache
from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES, LlamaAttention, LlamaDistributedAttention, LlamaPreTrainedModel


def create_distributed_llama(model_name_or_config):
    LLAMA_ATTENTION_CLASSES['eager'] = LlamaDistributedAttention
    model = AutoModelForCausalLM.from_pretrained(model_name_or_config, attn_implementation="eager")
    return model

def create_normal_llama(model_name_or_config):
    LLAMA_ATTENTION_CLASSES['eager'] = LlamaAttention
    model = AutoModelForCausalLM.from_pretrained(model_name_or_config, attn_implementation="eager")
    return model

def calculate_kv_cache(model, inputs):
    results = model.generate(inputs, max_new_tokens=1, return_dict_in_generate=True)
    cache = results['past_key_values']
    if not isinstance(cache, DistributedCache):
        raise TypeError("Model must be a distributed model")
    if not cache.get_seq_length() == inputs.shape[1]: # todo crop!
        raise ValueError("Wrong cache size")
    return cache

def calculate_kv_static_cache(model, inputs):
    results = model.generate(inputs, max_new_tokens=1, return_dict_in_generate=True)
    cache = results['past_key_values']
    if not isinstance(cache, DistributedCache):
        raise TypeError("Model must be a distributed model")
    if not cache.get_seq_length() == inputs.shape[1]: # todo crop!
        raise ValueError("Wrong cache size")
    
    new_cache = DistributedCache()
    new_cache.save_static_cache([cache.dynamic_cache.key_cache], [cache.dynamic_cache.value_cache])
    return new_cache

def generate(model: LlamaPreTrainedModel, inputs, cache=None, distribution_info=None, **kwargs):
    if not cache:
        return model.generate(inputs,  **kwargs)

    attention_mask = torch.ones([1, cache.get_seq_length() + inputs.shape[1]], dtype=torch.long)
    return model.generate(inputs, past_key_values=cache, attention_mask=attention_mask, **kwargs)
    