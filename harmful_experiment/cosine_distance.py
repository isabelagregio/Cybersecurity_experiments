from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Carregar modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_distance(vec1, vec2):
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def process_responses(data):
    chosen_embedding = model.encode(data["chosen"], convert_to_numpy=True)
    rejected_embedding = model.encode(data["rejected"], convert_to_numpy=True)
    
    distances = {}
    min_dist_to_chosen = float("inf")
    max_dist_to_rejected = float("-inf")
    corrected_response = ""
    
    for key, response in data["model_response"].items():
        response_embedding = model.encode(response, convert_to_numpy=True)
        
        dist_to_chosen = cosine_distance(response_embedding, chosen_embedding)
        dist_to_rejected = cosine_distance(response_embedding, rejected_embedding)
        
        distances[key] = {"distance_to_chosen": dist_to_chosen, "distance_to_rejected": dist_to_rejected}
        
        if dist_to_chosen < min_dist_to_chosen and dist_to_rejected > max_dist_to_rejected:
            min_dist_to_chosen = dist_to_chosen
            max_dist_to_rejected = dist_to_rejected
            corrected_response = response
    
    data["corrected_response"] = corrected_response
    return data, distances

# Importar JSON
with open("harmful_experiment_augmented_responses_secondpart.json", "r") as f:
    json_data = json.load(f)

processed_data_list = []
distances_list = []

for item in json_data:
    processed_item, distances = process_responses(item)
    processed_data_list.append(processed_item)
    distances_list.append(distances)

# Salvar em um novo JSON
with open("harmful_experiment_adjusted_response_secondpart.json", "w") as f:
    json.dump(processed_data_list, f, indent=4)

# Exibir distâncias calculadas
print(json.dumps(distances_list, indent=4))
