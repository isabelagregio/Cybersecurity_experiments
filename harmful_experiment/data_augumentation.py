import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuração do modelo
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token, torch_dtype=torch.float16).cuda()

def clean_response(response, original_text):
    """
    Processa a resposta do modelo para remover repetições e garantir variações válidas.
    """
    lines = response.strip().split("\n")

    # Filtra frases inválidas
    clean_lines = []
    for line in lines:
        lower_line = line.lower().strip()
        if (lower_line and 
            original_text.lower() not in lower_line and
            "rewrite the following" not in lower_line and
            "using synonyms while keeping its meaning" not in lower_line and
            "provide" not in lower_line and
            "rewritten" not in lower_line and 
            "synonyms" not in lower_line and
            "variation" not in lower_line):
            clean_lines.append(line.strip())

    # Se não houver variações válidas, repetir o original
    while len(clean_lines) < 2:
        clean_lines.append(original_text)

    return clean_lines[:2]


# Função para gerar variações com sinônimos
def generate_variations(original_text, num_variations=2):
    prompt = f"""
    Rewrite the following sentence using synonyms while keeping its meaning:
    "{original_text}"
    Provide {num_variations} variations.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=50, temperature=0.7, top_p=0.9, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Limpar a resposta para remover repetições da instrução
    variations = clean_response(response, original_text)

    return variations

# Função principal
def main():
    json_file = "response_harmful_mistral.json"

    # Carregar JSON
    with open(json_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    augmented_data = []

    for item in json_data[1999:]:
        prompt = item.get("prompt")
        rejected = item.get("rejected")
        chosen = item.get("chosen")
        original = item.get("model_response")
        variations = generate_variations(prompt, num_variations=2)
            
        new_item = {
                "prompts": {
                    "original": prompt,
                    "variation_1": variations[0],
                    "variation_2": variations[1]
                },
                "rejected": rejected,
                "chosen": chosen,
                "model_response": {
                    "original": original
                }
            }
        augmented_data.append(new_item)

    # Salvar novo JSON com as variações
    with open("harmful_experiment_augmented_secondpart.json", "w", encoding="utf-8") as file:
        json.dump(augmented_data, file, indent=4, ensure_ascii=False)

    print("Dataset aumentado salvo em 'harmful_experiment_augmented.json'.")

if __name__ == "__main__":
    main()
