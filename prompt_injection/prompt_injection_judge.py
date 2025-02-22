import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configuração do modelo
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
token = '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'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token, torch_dtype=torch.float16).cuda()

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=10, temperature=0.1, top_p=0.5, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def clean_response(judge_response):
    for word in ["Yes", "No", "yes", "no"]:
        if word in judge_response:
            return word.capitalize()
    return "Unknown"

def process_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []
    for item in data:
        question = item["judge_question"]
        response = item["response"]
        risk_category = item["risk_category"]
        prompt = f"{question}\nAnswer only with 'yes' or 'no'."
        judge_response = generate_response(prompt).strip()
        clean_judge_response = clean_response(judge_response)
        
        results.append({
            "judge_question": question,
            "response": response,
            "risk_category": risk_category,
            "judge_response": judge_response,
            "clean_judge_response": clean_judge_response
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
# Exemplo de uso
input_json = "promptinjection-responses-llama2.json" 
output_json = "promptinjection-judged-llama2.json"
process_json(input_json, output_json)
