from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caminho do modelo salvo
peft_model_path = "mistral_finetuned"
config = PeftConfig.from_pretrained(peft_model_path)

ds = load_dataset("LLM-LAT/benign-dataset")
split_ds = ds["train"].train_test_split(test_size=0.1)

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
# Carrega o modelo base novamente (o mesmo que usou no fine-tuning)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(base_model, peft_model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def create_prompt(sample):
    bos_token = "<s>"
    eos_token = "</s>"
    
    input_text = sample["prompt"].strip()
    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "### Prompt:"
    full_prompt += "\n" + input_text
    full_prompt += "\n\n### Response:"
    full_prompt += eos_token

    return full_prompt

def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')
  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded_output = tokenizer.batch_decode(generated_ids)
  return decoded_output[0].replace(prompt, "")

def process_dataset(dataset):
    results = []
    
    for sample in dataset:
        prompt = sample["prompt"]
        model_response = generate_response(prompt, model)
        response = sample["response"]
        refusal = sample["refusal"]
        result = {
            "prompt": prompt,
            "response": response,
            "refusal": refusal,
            "model_response": model_response
        }
        
        results.append(result)
        logger.info(f"Processed prompt: {prompt}")
    
    return results




processed_results = process_dataset(split_ds["test"].select(range(495)))
# Save the results to a JSON file

output_file = Path("response_bening_finetuned.json")
with open(output_file, "w") as f:
    json.dump(processed_results, f, indent=4)

logger.info(f"Results saved to {output_file} agora atualizado!!")

