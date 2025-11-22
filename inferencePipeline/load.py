import os
# 保持环境变量配置
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class MyModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_model()

    def get_model(self):
        model_name = 'Qwen/Qwen3-8B' # 用户已更新为 4B
        cache_dir = r'/app/models'
        print(f"Loading model {model_name} from {cache_dir}")

        # 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            padding_side='left' # 【关键点】生成任务必须使用左填充
        )
        
        # 确保 pad_token 存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir=cache_dir,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        self.model.eval()

    def _detect_complexity(self, question):
        """保持原本的复杂度检测逻辑"""
        question_lower = question.lower()
        simple_indicators = ['true or false', 'yes or no', 'is it', 'does it', 'capital of', 'what is the']
        if any(indicator in question_lower for indicator in simple_indicators):
            return 'simple'
        complex_indicators = ['explain', 'why', 'how does', 'compare', 'analyze', 'calculate', 'prove']
        if any(indicator in question_lower for indicator in complex_indicators):
            return 'complex'
        return 'medium'

    def _get_generation_config(self, complexity):
        """根据复杂度返回生成参数 - 增加 token 限制防止截断，降低温度提升稳定性"""
        if complexity == 'simple':
            return 96, 0.01, "Answer in one short sentence." # 增加到 96，严格指令
        elif complexity == 'complex':
            return 256, 0.2, "Answer clearly."
        else:
            return 128, 0.1, "Answer accurately."

    def _build_prompt(self, question, instruction):
        """构建 Prompt 字符串 - 回归 Q: A: 格式，配合严格后处理"""
        return f"""Role: You are a helpful assistant.
Instruction: {instruction}

Q: {question}
A:"""

    def __call__(self, questions):
        # 1. 对问题按复杂度分组
        batches = {'simple': [], 'medium': [], 'complex': []}
        
        # 记录原始顺序
        id_map = {q['questionID']: i for i, q in enumerate(questions)}
        final_results = [None] * len(questions)

        for q in questions:
            comp = self._detect_complexity(q['question'])
            batches[comp].append(q)

        # 2. 遍历每种复杂度，进行批处理推理
        for complexity, batch_questions in batches.items():
            if not batch_questions:
                continue
            
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Processing {len(batch_questions)} {complexity} questions...", flush=True)
            
            max_new_tokens, temperature, instruction = self._get_generation_config(complexity)
            
            # 构建 Prompt
            prompts = [self._build_prompt(q['question'], instruction) for q in batch_questions]
            
            try:
                # 批量 Tokenize
                inputs = self.tokenizer(
                    prompts, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True,
                    max_length=2048
                ).to(self.device)

                # 批量 Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3
                    )

                # 解码
                input_len = inputs.input_ids.shape[1]
                generated_tokens = outputs[:, input_len:]
                decoded_answers = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                # 后处理
                for q, raw_answer in zip(batch_questions, decoded_answers):
                    clean_answer = raw_answer.strip()
                    # 清理 think 标签
                    clean_answer = clean_answer.replace('<think>', '').replace('</think>', '')
                    
                    # 1. 移除 Q: A: 后续
                    # 使用 unicode escape \uff1a 代表全角冒号
                    clean_answer = re.split(r'\n\s*[QA][;:\uff1a]', clean_answer)[0].strip()
                    
                    # 2. 移除对话残留 (Wait, Okay, However, Now answer)
                    # 匹配换行后紧接这些词的情况
                    clean_answer = re.split(r'\n\s*(Wait|Okay|However|But|The answer is|Now answer)', clean_answer, flags=re.IGNORECASE)[0].strip()
                    
                    # 3. 移除双换行后的内容 (假设答案通常是一段，特别是简单问题)
                    if complexity == 'simple':
                         clean_answer = clean_answer.split('\n\n')[0].strip()

                    # 移除 JUDGE 相关的残留 (以防万一)
                    if '[JUDGE' in clean_answer:
                        clean_answer = clean_answer.split('[JUDGE')[0].strip()

                    # 按原始索引存入
                    index = id_map[q['questionID']]
                    final_results[index] = {'questionID': q['questionID'], 'answer': clean_answer}

            except Exception as e:
                print(f"Error batch processing {complexity}: {e}")
                for q in batch_questions:
                    index = id_map[q['questionID']]
                    final_results[index] = {'questionID': q['questionID'], 'answer': ''}

        return final_results

def loadPipeline():
    return MyModel()

if __name__ == '__main__':
    pipeline = loadPipeline()
    questions = [
        {'questionID': 1, 'question': 'what is the capital of Ireland?'}, 
        {'questionID': 2, 'question': 'what is the capital of Italy?'},
        {'questionID': 3, 'question': 'Is the earth flat? yes or no'},
        {'questionID': 4, 'question': 'Explain quantum physics briefly.'}
    ]
    answers = pipeline(questions)
    print(answers)