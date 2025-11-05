# save as check_env.py and run python check_env.py
import sys, torch
print('python', sys.version)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('num_threads', torch.get_num_threads())

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'  # Jiang: allow online HF downloads
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Jiang: allow online transformers fetch



from transformers import AutoTokenizer, AutoModelForCausalLM


class MyModel:

    def __init__(self):
        self.get_model()

    def __call__(self, questions):
        answers = []
        for q in questions:
            try:
                a = self.get_answer(q['question'])
            except:
                a = ''
            answers.append({'questionID': q['questionID'], 'answer': a})
        return answers

    def get_model(self):
        model_name = 'Qwen/Qwen3-0.6B' # Jiang: changed to Qwen3-4B
        print('Loading model:', model_name)  # Jiang: debug print

        # cache_dir = './app/models'  # Jiang: changed to relative path
        cache_dir = './app/models'  # Jiang: changed to relative path

        os.makedirs(cache_dir, exist_ok=True)  # Jiang: ensure cache dir exists

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name,
                                                  cache_dir=cache_dir)

        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                     cache_dir=cache_dir,
                                                     device_map='auto',
                                                     torch_dtype=torch.bfloat16) # Jiang: dtype changed to torch_dtype
        self.model = model
        self.tokenizer = tokenizer

    def get_answer(self, question):
        prompt = f'provide a brief answer to the following question: {question}'
 
        inputs = self.tokenizer.encode(prompt, padding=True, return_tensors='pt').to(self.model.device)

        attention_mask = torch.ones_like(inputs)

        outputs = self.model.generate(inputs,
                                      attention_mask=attention_mask,
                                      max_new_tokens=100,
                                      pad_token_id= self.tokenizer.eos_token_id)

        answer = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        print('Raw answer:', answer)  # Jiang: debug print

        return answer.strip().replace('\n', ' ')


def loadPipeline():
    return MyModel()


if __name__ == '__main__':

    pipeline = loadPipeline()

    questions = [{'questionID': 123, 'question': 'what is the capital of Ireland?'}, 
                 {'questionID': 456, 'question': 'what is the capital of Italy?'}]

    answers = pipeline(questions)

    print(answers)
