# import pickle
# import numpy as np
# import os
# from typing import List, Dict, Any, Tuple
# from dataclasses import dataclass
# from tqdm import tqdm
# from openai import OpenAI, AzureOpenAI
# from datasets import load_dataset
# from dataclasses import dataclass, asdict, field
# import logging
# import json
# import torch
# import torch.nn.functional as F
# import yaml
# import pandas as pd
# import re
# import concurrent.futures
# import tenacity
# import sys
# sys.path.append('../saerch')
# import family

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# torch.set_grad_enabled(False)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Set the subject here. Can be "astroPH" or "csLG"
# SUBJECT = "astroPH"

# # Subject-specific settings
# SUBJECT_SETTINGS = {
#     "astroPH": {
#         "inputs": ['astronomer', 'astronomy'],
#         "n_dirs": 9216,
#         "k": 64
#     },
#     "csLG": {
#         "inputs": ['machine learning researcher', 'machine learning'],
#         "n_dirs": 2048,
#         "k": 32
#     }
# }

# @tenacity.retry(
#         wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
#         stop=tenacity.stop_after_attempt(5),
#         retry=tenacity.retry_if_exception_type(Exception)
#     )
# def llm_prompt(prompt, client, model = 'gpt-35-turbo', return_text = False):
#     response = client.chat.completions.create(
#             model=model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#         )
#     response_text = response.choices[0].message.content
    
#     try:
#         prediction = response_text.split("FINAL:")[1].strip()
#     except Exception as e:
#         logging.error(f"Error summarizing family: {e}")
#         prediction = ""
    
#     if return_text:
#         return prediction, response_text

#     return prediction

# class EmbeddingClient:
#     def __init__(self, client: OpenAI, model: str = "text-embedding-3-small"):
#         self.client = client
#         self.model = model

#     def embed(self, text: str) -> np.ndarray:
#         embedding = self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
#         return np.array(embedding, dtype=np.float32)

#     def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
#         embeddings = self.client.embeddings.create(input=texts, model=self.model).data
#         return [np.array(embedding.embedding, dtype=np.float32) for embedding in embeddings]

# class Evaluator():
#     def __init__(self, subject, embeddings_path, texts_path):
#         logging.info(f"Initializing Evaluator for subject: {subject}")
#         self.subject = subject
#         self.abstract_embeddings = np.load(embeddings_path)
#         with open(texts_path, 'r') as f:
#             self.abstract_texts = json.load(f)
        
#         self.config = yaml.safe_load(open('../config.yaml'))
#         self.embedding_client = EmbeddingClient(OpenAI(api_key=self.config['jwu_openai_key']))
#         self.client = AzureOpenAI(
#             azure_endpoint=self.config["base_url"],
#             api_key=self.config["azure_api_key"],
#             api_version=self.config["api_version"],
#         )
#         self.upweight_ind = None
#         self.downweight_ind = None
        
#         self.inputs = SUBJECT_SETTINGS[subject]["inputs"]

#         logging.info("Evaluator initialized successfully")
        
#     def get_search_results(self, embedding):
#         logging.info("Getting search results")
#         if type(embedding) == torch.Tensor:
#             if embedding.requires_grad:
#                 embedding = embedding.detach()
#         sims = (self.abstract_embeddings @ embedding.numpy()).flatten()
#         topk_indices_search = np.argsort(sims)[::-1][:10]
        
#         result_str = ""
#         for i in topk_indices_search:
#             result_str += self.abstract_texts['abstracts'][i] + '\n\n'

#         logging.info(f"Retrieved {len(topk_indices_search)} search results")
        
#         return result_str

#     def compare_retrieval(self, query, model):
#         logging.info(f"Comparing retrieval for query: {query[:50]}...")
#         query_emb = torch.tensor(self.embedding_client.embed(query)).to(device)
#         query_act = model.ae(query_emb)
#         query_search_results = self.get_search_results(query_emb)

#         # Select features to upweight and downweight
#         upweight_label, downweight_label, llm_results, auto_results = self.adjust_query(query, query_emb, query_act, model)

#         # Generate multichoice options
#         mult_choice = self.generate_multichoice(upweight_label, downweight_label, model)

#         # Guess for query rewriting (LLM)
#         llm_guess = self.guess_retrieval(query_search_results, llm_results['up'], mult_choice)

#         # Guess for direct latent intervention (auto)
#         auto_guess = self.guess_retrieval(query_search_results, auto_results['up'], mult_choice)

#         logging.info("Retrieval comparison completed")

#         return {
#             'query': query,
#             'upweight_label': upweight_label,
#             'downweight_label': downweight_label,
#             'mult_choice': mult_choice,
#             'llm_guess': llm_guess,
#             'auto_guess': auto_guess,
#             'query_search_results': query_search_results,
#             'llm_results': llm_results,
#             'auto_results': auto_results
#         }
    
#     def adjust_query(self, query, query_emb, query_act, model):
#         logging.info("Adjusting query")

#         # Select a random feature to downweight from activating examples
#         query_topk = query_act[1]['topk_indices'].detach().numpy()
#         clean_ids = model.clean_labels_by_id.keys()
#         query_topk = [ind for ind in query_topk if ind in clean_ids]
#         self.downweight_ind = np.random.choice(query_topk, 1)[0]
#         downweight_label = model.clean_labels_by_id[self.downweight_ind]['label']

#         # Load the perfect indices
#         path = f"../saerch/sae_data_{self.subject}/feature_analysis_results_{SUBJECT_SETTINGS[self.subject]['k']}.json"

#         with open(path, 'r') as f:
#             feature_data = json.load(f)

#         # Get the indices of the features with greater than 0.99 pearson_correlation
#         high_corr_indices = [item['index'] for item in feature_data if item['pearson_correlation'] > 0.99]
#         logging.info(f"High correlation indices: {len(high_corr_indices)}")

#         # Select a random zero feature to upweight
#         zero_features = [ind for ind in clean_ids if ind not in query_topk]
#         logging.info(f"Zero features: {len(zero_features)}")
#         # Get intersection with high correlation indices
#         zero_features = list(set(zero_features).intersection(set(high_corr_indices)))
#         logging.info(f"Zero features after intersection: {len(zero_features)}")
#         self.upweight_ind = np.random.choice(zero_features, 1)[0]
#         upweight_label = model.clean_labels_by_id[self.upweight_ind]['label']
        
#         rewrite_prompt = f"""You are an expert {self.inputs[0]} trying to construct a query to answer your research questions. 
#                             Given the following query: {query}, you need to do the following:
#                             1. Downweights the importance of this specific topic: {downweight_label}
#                             2. Add this specific topic to the end: {upweight_label}. Add it as '<original query> and <new topic>'

#                             When adjusting the query to downweight the specific downweighting topic, you cannot just completely rewrite the query. Your query should return the same/
#                             results as the original query, just with the specific topic removed. Do NOT make your modified query too different from the original query.
#                             The goal of the query should be to decrease the impact of the first topic and increase the impact of the second topic on the search results.
#                             Return your answer in the format FINAL: <rewritten query>"""
        
#         llm_rewrite = llm_prompt(rewrite_prompt, self.client)
#         llm_rewrite_emb = torch.tensor(self.embedding_client.embed(llm_rewrite)).to(device)
#         llm_up_results = self.get_search_results(llm_rewrite_emb)

#         # Calculate cosine similarity for LLM rewrite
#         llm_cosine_sim = F.cosine_similarity(query_emb.unsqueeze(0), llm_rewrite_emb.unsqueeze(0)).item() #(query_emb @ llm_rewrite_emb).item()

#         logging.info(f"LLM rewritten query: {llm_rewrite[:50]}...")
#         logging.info(f"LLM upweighted results: {llm_up_results}")
#         logging.info(f"LLM rewrite cosine similarity: {llm_cosine_sim}")

#         # Automatic query adjustment
#         query_latents = query_act[1]['latents_pre_act'].detach().clone()

#         # Apply top-k selection
#         topk_values, topk_indices = torch.topk(query_latents, k=model.ae.k, dim=-1)
#         topk_values = F.relu(topk_values)  # Apply ReLU activation

#         # Create sparse representation
#         adjusted_latents = torch.zeros_like(query_latents)
#         adjusted_latents.scatter_(-1, topk_indices, topk_values)

#         # Adjust the specific features
#         adjusted_latents[self.downweight_ind] = 0  # Set downweighted feature to zero
#         adjusted_latents[self.upweight_ind] = 10  # Set upweighted feature to 10

#         # New topk indices and values from adjusted latents
#         topk_values_adjusted, topk_indices_adjusted = torch.topk(adjusted_latents, k=model.ae.k, dim=-1)

#         # Use the model's decode_sparse method
#         auto_adjusted_emb = model.ae.decode_sparse(topk_indices_adjusted, topk_values_adjusted)

#         # Calculate cosine similarity for auto adjustment
#         auto_cosine_sim = F.cosine_similarity(query_emb.unsqueeze(0), auto_adjusted_emb.unsqueeze(0)).item() #(query_emb @ auto_adjusted_emb).item()

#         # Log original topk indices and values
#         logging.info(f"Original topk indices: {topk_indices}")
#         logging.info(f"Original topk values: {topk_values}")
#         # Log adjusted topk indices and values
#         logging.info(f"Adjusted topk indices: {topk_indices_adjusted}")
#         logging.info(f"Adjusted topk values: {topk_values_adjusted}")
#         logging.info(f"Auto adjustment cosine similarity: {auto_cosine_sim}")

#         auto_up_results = self.get_search_results(auto_adjusted_emb)

#         logging.info("Automatic query adjustment completed")
#         logging.info(f"Auto upweighted results: {auto_up_results}")

#         return upweight_label, downweight_label, {
#             'up': llm_up_results,
#             'original_query': query,
#             'rewritten_query': llm_rewrite,
#             'cosine_similarity': llm_cosine_sim
#         }, {
#             'up': auto_up_results,
#             'cosine_similarity': auto_cosine_sim
#         }
    
#     def adjust_query_auto(self, query_emb, query_act, model, downweight_value, upweight_value):
#         query_latents = query_act[1]['latents_pre_act'].detach().clone()

#         topk_values, topk_indices = torch.topk(query_latents, k=model.ae.k, dim=-1)
#         topk_values = F.relu(topk_values)

#         adjusted_latents = torch.zeros_like(query_latents)
#         adjusted_latents.scatter_(-1, topk_indices, topk_values)

#         # Adjust the specific features using the provided values
#         adjusted_latents[self.downweight_ind] = downweight_value
#         adjusted_latents[self.upweight_ind] = upweight_value

#         topk_values_adjusted, topk_indices_adjusted = torch.topk(adjusted_latents, k=model.ae.k, dim=-1)

#         auto_adjusted_emb = model.ae.decode_sparse(topk_indices_adjusted, topk_values_adjusted)

#         auto_cosine_sim = F.cosine_similarity(query_emb.unsqueeze(0), auto_adjusted_emb.unsqueeze(0)).item() #(query_emb @ auto_adjusted_emb).item()

#         auto_up_results = self.get_search_results(auto_adjusted_emb)

#         return {
#             'up': auto_up_results,
#             'cosine_similarity': auto_cosine_sim
#         }

#     def generate_multichoice(self, upweight_label, downweight_label, model):
#         choices = [
#             upweight_label,
#             downweight_label,
#             model.clean_labels_by_id[np.random.choice(list(model.clean_labels_by_id.keys()))]['label'],
#             model.clean_labels_by_id[np.random.choice(list(model.clean_labels_by_id.keys()))]['label'],
#             model.clean_labels_by_id[np.random.choice(list(model.clean_labels_by_id.keys()))]['label']
#         ]
#         np.random.shuffle(choices)
#         return {i+1: choice for i, choice in enumerate(choices)}

#     def guess_retrieval(self, original_results, modified_results, mult_choice):
#         logging.info("Guessing retrieval")
        
#         guess_prompt = f"""You are an expert {self.inputs[0]} evaluating a retrieval system for research papers.
        
#                         INPUT DESCRIPTION:
#                         You will be given two sets of titles and abstracts corresponding to research papers retrieved for a query. 
#                         You will also be given 5 potential topics, numbered 1 to 5. The importance of exactly one of these topics has been upweighted and one has been downweighted between the first and second set of papers.
                        
#                         OUTPUT DESCRIPTION:
#                         Write out a list of specific {self.inputs[1]} topics covered by the first set, and another list of topics covered by the second set.
#                         Compare and contrast the two sets of search results. Based on the 5 potential topics, determine which topic has most likely been upweighted and which has been downweighted in the second set.
#                         Return your answer in the format FINAL: <upweighted number>, <downweighted number>

#                         ORIGINAL SEARCH RESULTS: {original_results}
#                         --------------------------------
#                         MODIFIED SEARCH RESULTS: {modified_results}
#                         --------------------------------
#                         POTENTIAL TOPICS: {mult_choice}
#                         """

#         guess, text = llm_prompt(guess_prompt, self.client, return_text=True, model='gpt-4o')

#         logging.info(f"Retrieval guess: {guess}")

#         return {'guess': guess, 'reasoning': text}


# # def save_results(results: Dict, filename):
# #     with open(filename, 'w') as f:
# #         json.dump(results, f, indent=2)

# def save_results(results: Dict, filename):
#     def convert_keys_to_str(obj):
#         if isinstance(obj, dict):
#             return {str(key): convert_keys_to_str(value) for key, value in obj.items()}
#         elif isinstance(obj, list):
#             return [convert_keys_to_str(element) for element in obj]
#         else:
#             return obj

#     serializable_results = convert_keys_to_str(results)
#     with open(filename, 'w') as f:
#         json.dump(serializable_results, f, indent=2)

# def load_results(filename):
#     with open(filename, 'r') as f:
#         return json.load(f)

# def parse_guess(guess_data):
#     if isinstance(guess_data, dict) and 'guess' in guess_data:
#         guess = guess_data['guess']
#     elif isinstance(guess_data, str):
#         guess = guess_data
#     else:
#         print(f"Unexpected guess data format: {guess_data}")
#         return None, None

#     guess = re.sub(r'\*', '', guess)
#     # Extract the first two numbers from the guess string
#     numbers = re.findall(r'\d+', guess)
#     if len(numbers) >= 2:
#         return int(numbers[0]), int(numbers[1])
#     else:
#         print(f"Error parsing guess: {guess}")
#         return None, None

# def main():
#     n_dirs = SUBJECT_SETTINGS[SUBJECT]["n_dirs"]
#     k = SUBJECT_SETTINGS[SUBJECT]["k"]
#     EVALUATION_RESULTS = f'evaluation_results_{SUBJECT}_{n_dirs}_{k}_multi_weights.json'
#     N = 50
#     SAE_DATA_DIR = f'../saerch/sae_data_{SUBJECT}/'
#     emb_path = f'../data/vector_store_{SUBJECT}/abstract_embeddings.npy'
#     texts_path = f'../data/vector_store_{SUBJECT}/abstract_texts.json'
    
#     embeddings = np.load(emb_path)
#     num_abstracts = embeddings.shape[0]
#     print(f"Loaded {num_abstracts} abstract embeddings for {SUBJECT}.")
#     del embeddings

#     logging.info("Starting main function")

#     evaluator = Evaluator(subject=SUBJECT, embeddings_path=emb_path, texts_path=texts_path)

#     model = family.Model(sae_data_dir=SAE_DATA_DIR, 
#                          model_path=f'../models/{k}_{n_dirs}_128_auxk_epoch_50.pth', 
#                          topk_indices=f'../saerch/sae_data_{SUBJECT}/topk_indices_{k}_{n_dirs}.npy',
#                          dataloader=None, 
#                          num_abstracts=num_abstracts,
#                          topk_values=f'../saerch/sae_data_{SUBJECT}/topk_values_{k}_{n_dirs}.npy', 
#                          autointerp_results=f'feature_analysis_results_{k}.json',
#                          mat=f'unnorm_cooccurrence_{k}_{n_dirs}.npy', 
#                          norms=f'occurrence_norms_{k}_{n_dirs}.npy', 
#                          actsims=f'actsims_{k}_{n_dirs}.npy')
    
#     logging.info("Evaluator and model initialized")

#     real_data = pd.read_csv('../clean_data.csv')
#     real_queries = real_data['full_user_query'].dropna()
#     cleaned_queries = real_queries.str.replace(r'<@U0.*?>', '', regex=True)
#     cleaned_queries = cleaned_queries.unique()
#     cleaned_queries = np.random.choice(cleaned_queries, N)
    
#     downweight_list = [-5, -1, -0.5, -0.25, 0]
#     upweight_list = [0.25, 0.5, 1, 5, 10]

#     all_results = {}

#     for query in tqdm(cleaned_queries[:2], desc="Processing queries"):
#         query_results = evaluator.compare_retrieval(query, model)
        
#         # Store query rewriting results (LLM)
#         all_results[query] = {
#             'llm_results': query_results['llm_results'],
#             'upweight_label': query_results['upweight_label'],
#             'downweight_label': query_results['downweight_label'],
#             'mult_choice': query_results['mult_choice'],
#             'auto_results': {}
#         }

#         query_emb = torch.tensor(evaluator.embedding_client.embed(query)).to(device)
#         query_act = model.ae(query_emb)

#         # Run direct latent intervention for all combinations
#         for down_val in downweight_list:
#             for up_val in upweight_list:
#                 auto_results = evaluator.adjust_query_auto(query_emb, query_act, model, down_val, up_val)
#                 all_results[query]['auto_results'][f"{down_val},{up_val}"] = auto_results

#     save_results(all_results, EVALUATION_RESULTS)
#     print(f"Evaluation complete. Results saved to {EVALUATION_RESULTS}")

# def calculate_accuracy(results):
#     stats = {}
#     for down_val in [-5, -1, -0.5, -0.25, 0]:
#         for up_val in [0.25, 0.5, 1, 5, 10]:
#             key = f"{down_val},{up_val}"
#             stats[key] = {
#                 'llm_up_acc': 0,
#                 'llm_down_acc': 0,
#                 'auto_up_acc': 0,
#                 'auto_down_acc': 0,
#                 'llm_cosine_similarities': [],
#                 'auto_cosine_similarities': [],
#                 'total': 0
#             }

#     for query, result in results.items():
#         mult_choice = result['mult_choice']
#         upweight_label = result['upweight_label']
#         downweight_label = result['downweight_label']

#         correct_up = next(k for k, v in mult_choice.items() if v == upweight_label)
#         correct_down = next(k for k, v in mult_choice.items() if v == downweight_label)

#         # Handle LLM results
#         llm_guess = result.get('llm_guess', {})
#         llm_up, llm_down = parse_guess(llm_guess)

#         for key in stats:
#             stats[key]['total'] += 1
#             if llm_up is not None and llm_down is not None:
#                 stats[key]['llm_up_acc'] += (int(llm_up) == int(correct_up))
#                 stats[key]['llm_down_acc'] += (int(llm_down) == int(correct_down))
            
#             llm_results = result.get('llm_results', {})
#             if isinstance(llm_results, dict) and 'cosine_similarity' in llm_results:
#                 stats[key]['llm_cosine_similarities'].append(llm_results['cosine_similarity'])

#             auto_results = result['auto_results'].get(key, {})
#             auto_up, auto_down = parse_guess(auto_results.get('guess', ''))
#             if auto_up is not None and auto_down is not None:
#                 stats[key]['auto_up_acc'] += (int(auto_up) == int(correct_up))
#                 stats[key]['auto_down_acc'] += (int(auto_down) == int(correct_down))
            
#             if 'cosine_similarity' in auto_results:
#                 stats[key]['auto_cosine_similarities'].append(auto_results['cosine_similarity'])

#     for key in stats:
#         total = stats[key]['total']
#         if total > 0:
#             stats[key]['llm_up_acc'] /= total
#             stats[key]['llm_down_acc'] /= total
#             stats[key]['auto_up_acc'] /= total
#             stats[key]['auto_down_acc'] /= total
#         if stats[key]['llm_cosine_similarities']:
#             stats[key]['llm_cosine_avg'] = np.mean(stats[key]['llm_cosine_similarities'])
#         if stats[key]['auto_cosine_similarities']:
#             stats[key]['auto_cosine_avg'] = np.mean(stats[key]['auto_cosine_similarities'])

#     return stats

# if __name__ == "__main__":
#     #main()
#     results = load_results(f'evaluation_results_{SUBJECT}_{SUBJECT_SETTINGS[SUBJECT]["n_dirs"]}_{SUBJECT_SETTINGS[SUBJECT]["k"]}_multi_weights.json')
#     stats = calculate_accuracy(results)
#     save_results(stats, f'aggregated_results_{SUBJECT}_{SUBJECT_SETTINGS[SUBJECT]["n_dirs"]}_{SUBJECT_SETTINGS[SUBJECT]["k"]}_multi_weights.json')
#     print("Aggregated results saved.")


import pickle
import numpy as np
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
from datasets import load_dataset
from dataclasses import dataclass, asdict, field
import logging
import json
import torch
import torch.nn.functional as F
import yaml
import pandas as pd
import re
import concurrent.futures
import tenacity
import sys
sys.path.append('../saerch')
import family
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the subject here. Can be "astroPH" or "csLG"
SUBJECT = "astroPH"

# Subject-specific settings
SUBJECT_SETTINGS = {
    "astroPH": {
        "inputs": ['astronomer', 'astronomy'],
        "n_dirs": 9216,
        "k": 64
    },
    "csLG": {
        "inputs": ['machine learning researcher', 'machine learning'],
        "n_dirs": 2048,
        "k": 32
    }
}

@tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type(Exception)
    )
def llm_prompt(prompt, client, model = 'gpt-35-turbo', return_text = False):
    response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    response_text = response.choices[0].message.content
    
    try:
        prediction = response_text.split("FINAL:")[1].strip()
    except Exception as e:
        logging.error(f"Error summarizing family: {e}")
        prediction = ""
    
    if return_text:
        return prediction, response_text

    return prediction

class EmbeddingClient:
    def __init__(self, client: OpenAI, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        embedding = self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
        return np.array(embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = self.client.embeddings.create(input=texts, model=self.model).data
        return [np.array(embedding.embedding, dtype=np.float32) for embedding in embeddings]

class Evaluator():
    def __init__(self, subject, embeddings_path, texts_path):
        logging.info(f"Initializing Evaluator for subject: {subject}")
        self.subject = subject
        self.abstract_embeddings = np.load(embeddings_path)
        with open(texts_path, 'r') as f:
            self.abstract_texts = json.load(f)
        
        self.config = yaml.safe_load(open('../config.yaml'))
        self.embedding_client = EmbeddingClient(OpenAI(api_key=self.config['jwu_openai_key']))
        self.client = AzureOpenAI(
            azure_endpoint=self.config["base_url"],
            api_key=self.config["azure_api_key"],
            api_version=self.config["api_version"],
        )
        
        self.inputs = SUBJECT_SETTINGS[subject]["inputs"]

        logging.info("Evaluator initialized successfully")
        
    def get_search_results(self, embedding):
        logging.info("Getting search results")
        if type(embedding) == torch.Tensor:
            if embedding.requires_grad:
                embedding = embedding.detach()
        sims = (self.abstract_embeddings @ embedding.numpy()).flatten()
        topk_indices_search = np.argsort(sims)[::-1][:10]
        
        result_str = ""
        for i in topk_indices_search:
            result_str += self.abstract_texts['abstracts'][i] + '\n\n'

        logging.info(f"Retrieved {len(topk_indices_search)} search results")
        
        return result_str

    def compare_retrieval(self, query, model, downweight_val, upweight_val):
        logging.info(f"Comparing retrieval for query: {query[:50]}...")
        query_emb = torch.tensor(self.embedding_client.embed(query)).to(device)
        query_act = model.ae(query_emb)
        query_search_results = self.get_search_results(query_emb)

        # Select features to upweight and downweight
        upweight_label, downweight_label, llm_results, auto_results = self.adjust_query(query, query_emb, query_act, model, downweight_val, upweight_val)

        # Generate multichoice options
        mult_choice = self.generate_multichoice(upweight_label, downweight_label, model)

        # Guess for query rewriting (LLM)
        llm_guess = self.guess_retrieval(query_search_results, llm_results['up'], mult_choice)

        # Guess for direct latent intervention (auto)
        auto_guess = self.guess_retrieval(query_search_results, auto_results['up'], mult_choice)

        logging.info("Retrieval comparison completed")

        return {
            'query': query,
            'upweight_label': upweight_label,
            'downweight_label': downweight_label,
            'mult_choice': mult_choice,
            'llm_guess': llm_guess,
            'auto_guess': auto_guess,
            'query_search_results': query_search_results,
            'llm_results': llm_results,
            'auto_results': auto_results
        }
    
    def adjust_query(self, query, query_emb, query_act, model, downweight_val, upweight_val):
        logging.info("Adjusting query")

        # Select a random feature to downweight from activating examples
        query_topk = query_act[1]['topk_indices'].detach().numpy()
        clean_ids = model.clean_labels_by_id.keys()
        query_topk = [ind for ind in query_topk if ind in clean_ids]
        downweight_ind = np.random.choice(query_topk, 1)[0]
        downweight_label = model.clean_labels_by_id[downweight_ind]['label']

        # Load the perfect indices
        path = f"../saerch/sae_data_{self.subject}/feature_analysis_results_{SUBJECT_SETTINGS[self.subject]['k']}.json"

        with open(path, 'r') as f:
            feature_data = json.load(f)

        # Get the indices of the features with greater than 0.99 pearson_correlation
        high_corr_indices = [item['index'] for item in feature_data if item['pearson_correlation'] > 0.99]
        logging.info(f"High correlation indices: {len(high_corr_indices)}")

        # Select a random zero feature to upweight
        zero_features = [ind for ind in clean_ids if ind not in query_topk]
        logging.info(f"Zero features: {len(zero_features)}")
        # Get intersection with high correlation indices
        zero_features = list(set(zero_features).intersection(set(high_corr_indices)))
        logging.info(f"Zero features after intersection: {len(zero_features)}")
        upweight_ind = np.random.choice(zero_features, 1)[0]
        upweight_label = model.clean_labels_by_id[upweight_ind]['label']
        
        rewrite_prompt = f"""You are an expert {self.inputs[0]} trying to construct a query to answer your research questions. 
                            Given the following query: {query}, you need to do the following:
                            1. Downweights the importance of this specific topic: {downweight_label}
                            2. Add this specific topic to the end: {upweight_label}. Add it as '<original query> and <new topic>'

                            When adjusting the query to downweight the specific downweighting topic, you cannot just completely rewrite the query. Your query should return the same/
                            results as the original query, just with the specific topic removed. Do NOT make your modified query too different from the original query.
                            The goal of the query should be to decrease the impact of the first topic and increase the impact of the second topic on the search results.
                            Return your answer in the format FINAL: <rewritten query>"""
        
        llm_rewrite = llm_prompt(rewrite_prompt, self.client)
        llm_rewrite_emb = torch.tensor(self.embedding_client.embed(llm_rewrite)).to(device)
        llm_up_results = self.get_search_results(llm_rewrite_emb)

        # Calculate cosine similarity for LLM rewrite
        llm_cosine_sim = F.cosine_similarity(query_emb.unsqueeze(0), llm_rewrite_emb.unsqueeze(0)).item() #(query_emb @ llm_rewrite_emb).item()

        logging.info(f"LLM rewritten query: {llm_rewrite[:50]}...")
        logging.info(f"LLM upweighted results: {llm_up_results}")
        logging.info(f"LLM rewrite cosine similarity: {llm_cosine_sim}")

        # Automatic query adjustment
        query_latents = query_act[1]['latents_pre_act'].detach().clone()

        # Apply top-k selection
        topk_values, topk_indices = torch.topk(query_latents, k=model.ae.k, dim=-1)
        topk_values = F.relu(topk_values)  # Apply ReLU activation

        # Create sparse representation
        adjusted_latents = torch.zeros_like(query_latents)
        adjusted_latents.scatter_(-1, topk_indices, topk_values)

        # Adjust the specific features
        adjusted_latents[downweight_ind] = downweight_val  # Set downweighted feature
        adjusted_latents[upweight_ind] = upweight_val # Set upweighted feature

        # New topk indices and values from adjusted latents
        topk_values_adjusted, topk_indices_adjusted = torch.topk(adjusted_latents, k=model.ae.k, dim=-1)

        # Use the model's decode_sparse method
        auto_adjusted_emb = model.ae.decode_sparse(topk_indices_adjusted, topk_values_adjusted)

        # Calculate cosine similarity for auto adjustment
        auto_cosine_sim = F.cosine_similarity(query_emb.unsqueeze(0), auto_adjusted_emb.unsqueeze(0)).item() #(query_emb @ auto_adjusted_emb).item()

        # Log original topk indices and values
        logging.info(f"Original topk indices: {topk_indices}")
        logging.info(f"Original topk values: {topk_values}")
        # Log adjusted topk indices and values
        logging.info(f"Adjusted topk indices: {topk_indices_adjusted}")
        logging.info(f"Adjusted topk values: {topk_values_adjusted}")
        logging.info(f"Auto adjustment cosine similarity: {auto_cosine_sim}")

        auto_up_results = self.get_search_results(auto_adjusted_emb)

        logging.info("Automatic query adjustment completed")
        logging.info(f"Auto upweighted results: {auto_up_results}")

        return upweight_label, downweight_label, {
            'up': llm_up_results,
            'original_query': query,
            'rewritten_query': llm_rewrite,
            'cosine_similarity': llm_cosine_sim
        }, {
            'up': auto_up_results,
            'cosine_similarity': auto_cosine_sim
        }

    def generate_multichoice(self, upweight_label, downweight_label, model):
        choices = [
            upweight_label,
            downweight_label,
            model.clean_labels_by_id[np.random.choice(list(model.clean_labels_by_id.keys()))]['label'],
            model.clean_labels_by_id[np.random.choice(list(model.clean_labels_by_id.keys()))]['label'],
            model.clean_labels_by_id[np.random.choice(list(model.clean_labels_by_id.keys()))]['label']
        ]
        np.random.shuffle(choices)
        return {i+1: choice for i, choice in enumerate(choices)}

    def guess_retrieval(self, original_results, modified_results, mult_choice):
        logging.info("Guessing retrieval")
        
        guess_prompt = f"""You are an expert {self.inputs[0]} evaluating a retrieval system for research papers.
        
                        INPUT DESCRIPTION:
                        You will be given two sets of titles and abstracts corresponding to research papers retrieved for a query. 
                        You will also be given 5 potential topics, numbered 1 to 5. The importance of exactly one of these topics has been upweighted and one has been downweighted between the first and second set of papers.
                        
                        OUTPUT DESCRIPTION:
                        Write out a list of specific {self.inputs[1]} topics covered by the first set, and another list of topics covered by the second set.
                        Compare and contrast the two sets of search results. Based on the 5 potential topics, determine which topic has most likely been upweighted and which has been downweighted in the second set.
                        Return your answer in the format FINAL: <upweighted number>, <downweighted number>

                        ORIGINAL SEARCH RESULTS: {original_results}
                        --------------------------------
                        MODIFIED SEARCH RESULTS: {modified_results}
                        --------------------------------
                        POTENTIAL TOPICS: {mult_choice}
                        """

        guess, text = llm_prompt(guess_prompt, self.client, return_text = True, model = 'gpt-4o')

        logging.info(f"Retrieval guess: {guess}")

        return {'guess': guess, 'reasoning': text}

def save_results(results: List[Dict], filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def parse_guess(guess):
    guess = re.sub(r'\*', '', guess['guess'])
    # Extract the first two numbers from the guess string
    numbers = re.findall(r'\d+', guess)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    else:
        print(f"Error parsing guess: {guess}")
        return None, None

def calculate_accuracy(results):
    llm_correct_up = 0
    llm_correct_down = 0
    auto_correct_up = 0
    auto_correct_down = 0
    total = 0

    llm_up_distribution = Counter()
    llm_down_distribution = Counter()
    auto_up_distribution = Counter()
    auto_down_distribution = Counter()

    llm_cosine_similarities = []
    auto_cosine_similarities = []

    for result in results:
        total += 1
        mult_choice = result['mult_choice']
        upweight_label = result['upweight_label']
        downweight_label = result['downweight_label']

        # Find the correct indices
        correct_up = next(k for k, v in mult_choice.items() if v == upweight_label)
        correct_down = next(k for k, v in mult_choice.items() if v == downweight_label)
        print(f"Correct Up: {correct_up}, Correct Down: {correct_down}")

        # LLM guesses
        llm_up, llm_down = parse_guess(result['llm_guess'])
        print(f"LLM Up = {llm_up}, LLM Down = {llm_down}")
        if llm_up is not None and llm_down is not None:
            llm_correct_up += (int(llm_up) == int(correct_up))
            llm_correct_down += (int(llm_down) == int(correct_down))
            llm_up_distribution[int(llm_up)] += 1
            llm_down_distribution[int(llm_down)] += 1

        # Auto guesses
        auto_up, auto_down = parse_guess(result['auto_guess'])
        print(f"Auto Up = {auto_up}, Auto Down = {auto_down}")
        if auto_up is not None and auto_down is not None:
            auto_correct_up += (int(auto_up) == int(correct_up))
            auto_correct_down += (int(auto_down) == int(correct_down))
            auto_up_distribution[int(auto_up)] += 1
            auto_down_distribution[int(auto_down)] += 1

        # Cosine similarities
        llm_cosine_similarities.append(result['llm_results']['cosine_similarity'])
        auto_cosine_similarities.append(result['auto_results']['cosine_similarity'])

    return {
        'llm_up_acc': llm_correct_up / total,
        'llm_down_acc': llm_correct_down / total,
        'auto_up_acc': auto_correct_up / total,
        'auto_down_acc': auto_correct_down / total,
        'total': total,
        'llm_up_dist': llm_up_distribution,
        'llm_down_dist': llm_down_distribution,
        'auto_up_dist': auto_up_distribution,
        'auto_down_dist': auto_down_distribution,
        'llm_cosine_similarities': llm_cosine_similarities,
        'auto_cosine_similarities': auto_cosine_similarities
    }

def print_stats(stats):
    print("Accuracy Statistics:")
    print(f"Total queries processed: {stats['total']}")
    print("\nQuery Rewriting (LLM) Results:")
    print(f"Upweighting accuracy: {stats['llm_up_acc']:.2%}")
    print(f"Downweighting accuracy: {stats['llm_down_acc']:.2%}")
    print(f"Upweighting distribution: {dict(stats['llm_up_dist'])}")
    print(f"Downweighting distribution: {dict(stats['llm_down_dist'])}")
    print(f"Average cosine similarity: {np.mean(stats['llm_cosine_similarities']):.4f}")
    
    print("\nDirect Latent Intervention (Auto) Results:")
    print(f"Upweighting accuracy: {stats['auto_up_acc']:.2%}")
    print(f"Downweighting accuracy: {stats['auto_down_acc']:.2%}")
    print(f"Upweighting distribution: {dict(stats['auto_up_dist'])}")
    print(f"Downweighting distribution: {dict(stats['auto_down_dist'])}")
    print(f"Average cosine similarity: {np.mean(stats['auto_cosine_similarities']):.4f}")

def main():
    n_dirs = SUBJECT_SETTINGS[SUBJECT]["n_dirs"]
    k = SUBJECT_SETTINGS[SUBJECT]["k"]
    EVALUATION_RESULTS = f'evaluation_results_{SUBJECT}_{n_dirs}_{k}.json'
    N = 2 #50
    SAE_DATA_DIR = f'../saerch/sae_data_{SUBJECT}/'
    SAVE_INTERVAL = 5
    emb_path = f'../data/vector_store_{SUBJECT}/abstract_embeddings.npy'
    texts_path = f'../data/vector_store_{SUBJECT}/abstract_texts.json'
    embeddings = np.load(emb_path)
    num_abstracts = embeddings.shape[0]
    print(f"Loaded {num_abstracts} abstract embeddings for {SUBJECT}.")
    del embeddings

    logging.info("Starting main function")

    evaluator = Evaluator(subject=SUBJECT, embeddings_path=emb_path, texts_path=texts_path)

    logging.info(f"Loaded {num_abstracts} abstract embeddings for {SUBJECT}.")

    # load pre-trained model
    model = family.Model(sae_data_dir=SAE_DATA_DIR, 
                         model_path=f'../models/{k}_{n_dirs}_128_auxk_epoch_50.pth', 
                         topk_indices=f'../saerch/sae_data_{SUBJECT}/topk_indices_{k}_{n_dirs}.npy',
                         dataloader=None, 
                         num_abstracts=num_abstracts,
                         topk_values=f'../saerch/sae_data_{SUBJECT}/topk_values_{k}_{n_dirs}.npy', 
                         autointerp_results=f'feature_analysis_results_{k}.json',
                         mat=f'unnorm_cooccurrence_{k}_{n_dirs}.npy', 
                         norms=f'occurrence_norms_{k}_{n_dirs}.npy', 
                         actsims=f'actsims_{k}_{n_dirs}.npy')
    
    logging.info("Evaluator and model initialized")

    real_data = pd.read_csv('../clean_data.csv')
    real_queries = real_data['full_user_query'].dropna()
    cleaned_queries = real_queries.str.replace(r'<@U0.*?>', '', regex=True)
    cleaned_queries = cleaned_queries.unique()
    cleaned_queries = np.random.choice(cleaned_queries, N)
    
    # Perform query rewriting once
    evaluation_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {executor.submit(evaluator.compare_retrieval, query, model, 0, 0): i 
                           for i, query in enumerate(cleaned_queries)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                           total=len(cleaned_queries), 
                           desc="Evaluating queries"):
            feature_index = future_to_index[future]
            try:
                result = future.result()
                evaluation_results.append(result)
            
                if len(evaluation_results) % SAVE_INTERVAL == 0:
                    save_results(evaluation_results, EVALUATION_RESULTS)
                    print(f"Checkpoint saved. Processed {len(evaluation_results)} queries.")
                    
            except Exception as exc:
                print(f"Query {feature_index} generated an exception: {exc}")

    save_results(evaluation_results, EVALUATION_RESULTS)
    print(f"Evaluation complete. Results saved to {EVALUATION_RESULTS}")

    # Load evaluation results for further processing
    evaluation_results = json.load(open(EVALUATION_RESULTS))

    downweight_list = [-5, -1, -0.5, -0.25, 0]
    upweight_list = [0.25, 0.5, 1, 5, 10]
    
    combined_results = []

    for downweight_val in downweight_list:
        for upweight_val in upweight_list:
            logging.info(f"Evaluating for downweight={downweight_val}, upweight={upweight_val}")
            latent_results = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_index = {executor.submit(evaluator.compare_retrieval, result['query'], model, downweight_val, upweight_val): i 
                                   for i, result in enumerate(evaluation_results)}
                
                for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                                   total=len(evaluation_results), 
                                   desc=f"Evaluating latent adjustments (downweight={downweight_val}, upweight={upweight_val})"):
                    feature_index = future_to_index[future]
                    try:
                        result = future.result()
                        latent_results.append(result)
                    
                        if len(latent_results) % SAVE_INTERVAL == 0:
                            save_results(latent_results, f'latent_results_{downweight_val}_{upweight_val}.json')
                            print(f"Checkpoint saved for latent adjustments (downweight={downweight_val}, upweight={upweight_val}). Processed {len(latent_results)} queries.")
                            
                    except Exception as exc:
                        print(f"Query {feature_index} generated an exception: {exc}")

            save_results(latent_results, f'latent_results_{downweight_val}_{upweight_val}.json')
            print(f"Latent adjustment evaluation complete for downweight={downweight_val}, upweight={upweight_val}. Results saved.")

            combined_results.extend(latent_results)

    # Save all combined results
    save_results(combined_results, 'all_combined_results.json')
    print("All combined results saved to 'all_combined_results.json'.")

if __name__ == "__main__":
    main()