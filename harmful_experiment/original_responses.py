from datasets import load_dataset
from collections import defaultdict
from typing import Dict, Optional
import json
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
token = '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'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token, torch_dtype=torch.float16).cuda()
ds = load_dataset("LLM-LAT/harmful-dataset")

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define function to generate responses
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=70, temperature=0.1, top_p=0.5, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define the processing logic
def process_dataset(dataset):
    results = []
    
    for sample in dataset:
        prompt = sample["prompt"]
        # Generate model response
        model_response = generate_response(prompt)
        
        # Check if response is harmful or rejected
        rejected = sample["rejected"]
        chosen = sample["chosen"]
    
        # Prepare output data
        result = {
            "prompt": prompt,
            "rejected": rejected,
            "chosen": chosen,
            "model_response": model_response
        }
        
        results.append(result)
        
        # Log progress
        logger.info(f"Processed prompt: {prompt}")
    
    return results

# Process the dataset
processed_results = process_dataset(ds["train"])

# Save the results to a JSON file
output_file = Path("response_harmful_mistral.json")
with open(output_file, "w") as f:
    json.dump(processed_results, f, indent=4)

logger.info(f"Results saved to {output_file}")
