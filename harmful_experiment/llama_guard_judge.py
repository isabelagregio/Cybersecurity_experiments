from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Configuração do modelo
model_id = "meta-llama/Llama-Guard-3-1B"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Carregar o dataset JSON
with open("harmful_experiment_adjusted_response_secondpart.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Processar cada entrada no dataset
for entry in data:
    prompts = entry["prompts"]
    prompt_original = prompts["original"]

    model_responses = entry["model_response"]

    original_response = model_responses["original"]
    corrected_response = entry.get("corrected_response", "")

    classifications = {}
    
    for response_type, response in zip(["original", "corrected"], [original_response, corrected_response]):
        # Criar a conversa no formato esperado pelo modelo
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": prompt_original}]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]

        # Tokenizar a conversa
        input_ids = tokenizer.apply_chat_template(
            conversation, return_tensors="pt"
        ).to(model.device)

        prompt_len = input_ids.shape[1]

        # Gerar resposta do modelo
        output = model.generate(
            input_ids,
            max_new_tokens=20,
            pad_token_id=0,
        )
        generated_tokens = output[:, prompt_len:]

        # Decodificar resposta
        classification_result = tokenizer.decode(generated_tokens[0]).strip()
        classifications[response_type] = classification_result
    
    # Salvar no dataset
    entry.setdefault("classification", {})  
    entry["classification"].update(classifications)

# Salvar o dataset atualizado
with open("harmful_experiment_judged_secondpart.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)

print("Classificação concluída e salva em 'harmful_experiment_judged_secondpart.json'.")
