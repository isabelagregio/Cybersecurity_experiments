from collections import defaultdict
from typing import Dict, Optional
import json
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/CodeLlama-7b-Instruct-hf"
token = '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'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token, torch_dtype=torch.float16).cuda()

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=10, temperature=0.1, top_p=0.5, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    logging.basicConfig(level=logging.INFO)
    
    prompt_path = Path("./promptinjection.json")  
    response_path = Path("./responses-promptinjection-codellama13b.json")  
    
    logging.info("Início da execução")
    process_prompts(prompt_path, response_path)
    logging.info("Fim da execução")

def process_prompts(prompt_path: Path, response_path: Path) -> None:
    """
    Processa um dataset de prompts enviando-os ao LLM e salvando as respostas em um arquivo.
    """
    response_result = []
    with open(prompt_path, 'r') as file:
        prompts = json.load(file)
        for prompt in prompts:
            full_prompt = f"{prompt['test_case_prompt']}\n\nUser Input: {prompt['user_input']}"
            logging.info(f"Consultando para o prompt: {prompt['test_case_prompt']}")
            response = generate_response(full_prompt)
            response_result.append({
                "test_case_prompt": prompt['test_case_prompt'],
                "user_input": prompt['user_input'],
                "response": response,
                "judge_question": prompt['judge_question'],
                "risk_category": prompt['risk_category']
            })
    
    response_path.write_text(json.dumps(response_result, indent=4))

if __name__ == "__main__":
    main()
