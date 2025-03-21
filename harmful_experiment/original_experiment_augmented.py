from collections import defaultdict
from typing import Dict, Optional
import json
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
token = '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'

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token, torch_dtype=torch.float16).cuda()

def generate_response(prompt: Optional[str]) -> str:
    """Gera uma resposta para um prompt dado, verificando se ele não é None."""
    if not prompt:
        return "N/A"  # Retorna uma resposta padrão se o prompt for None
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=70, temperature=0.1, top_p=0.5, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    logging.basicConfig(level=logging.INFO)
    
    prompt_path = Path("./harmful_experiment_augmented_secondpart.json")  
    response_path = Path("./harmful_experiment_augmented_responses_secondpart.json")  
    
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
            original_prompt = prompt.get("prompts", {}).get("original")
            variation_1 = prompt.get("prompts", {}).get("variation_1")
            variation_2 = prompt.get("prompts", {}).get("variation_2")
            rejected = prompt.get("rejected")
            chosen = prompt.get("chosen")
            original = prompt.get("model_response", {}).get("original")

            response_variation_1 = generate_response(variation_1)
            response_variation_2 = generate_response(variation_2)

            response_result.append({
                "prompts": {
                    "original": original_prompt,
                    "variation_1": variation_1,
                    "variation_2": variation_2
                },
                "rejected": rejected,
                "chosen": chosen,
                "model_response": {
                    "original": original,
                    "variation_1": response_variation_1,
                    "variation_2": response_variation_2
                }
            })
    
    response_path.write_text(json.dumps(response_result, indent=4))

if __name__ == "__main__":
    main()
