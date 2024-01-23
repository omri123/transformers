import torch
import unittest
from textwrap import dedent
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.dist_llm_api import create_distributed_llama, create_normal_llama, calculate_kv_cache, generate, calculate_kv_static_cache



class DistributedLlamaTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DistributedLlamaTests, self).__init__(*args, **kwargs)
        self.device = "cpu"
        self.model_name = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.prompt = 'Genesis 1:1 In the beginning God created the' # 12 tokens
        self.response = 'heavens and the earth' # 5 tokens
    
    def convert_text_to_ids(self, text):
        return torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))).unsqueeze(0)

        
    def test_generate_from_normal_llama(self):
        model = create_normal_llama(self.model_name)
        encodeds = self.convert_text_to_ids(self.prompt)
        generated_ids = generate(model, encodeds, max_new_tokens=5, do_sample=False)
        decoded = self.tokenizer.batch_decode(generated_ids[:, encodeds.shape[1]:])
        self.assertEqual(decoded[0], self.response)
    
    def test_generate_from_distributed_llama(self):
        model = create_distributed_llama(self.model_name)
        encodeds = self.convert_text_to_ids(self.prompt)
        generated_ids = generate(model, encodeds, max_new_tokens=5, do_sample=False)
        decoded = self.tokenizer.batch_decode(generated_ids[:, encodeds.shape[1]:])
        self.assertEqual(decoded[0], self.response)
    
    def test_generate_with_cache(self):
        model = create_distributed_llama(self.model_name)
        encodeds = self.convert_text_to_ids(self.prompt)
        cache = calculate_kv_cache(model, encodeds)
        generated_ids = generate(model, self.convert_text_to_ids('heavens'), cache, max_new_tokens=3, do_sample=False)
        decoded = self.tokenizer.batch_decode(generated_ids)
        self.assertEqual(decoded[0], self.response)
    
    # def test_generate_with_long_cache(self): # Maybe just long context?
    #     pass
    
    # def test_generate_with_distributed_cache(self):
    #     pass
    
    # def test_generate_with_long_distributed_cache(self):
    #     pass
    
if __name__ == '__main__':
    unittest.main()
