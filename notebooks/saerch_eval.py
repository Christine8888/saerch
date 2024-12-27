import pickle
import numpy as np
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from dataclasses import dataclass, asdict, field
import logging
import json
import torch
import yaml
import pandas as pd
import re
import concurrent.futures
import tenacity
import sys
sys.path.append('../saerch')
import family

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type(Exception)
    )
def llm_prompt(prompt, client, model = 'gpt-3.5-turbo', return_text = False):
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
        # return np.zeros(1536, dtype = np.float32)
        embedding = self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
        return np.array(embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = self.client.embeddings.create(input=texts, model=self.model).data
        return [np.array(embedding.embedding, dtype=np.float32) for embedding in embeddings]

class Evaluator():
    def __init__(self, subject = 'astroPH', embeddings_path = '../data/vector_store_astroPH/abstract_embeddings.npy', texts_path = '../data/vector_store_astroPH/abstract_texts.json'):
        self.abstract_embeddings = np.load(embeddings_path)
        with open(texts_path, 'r') as f:
            self.abstract_texts = json.load(f)
        
        self.config = yaml.safe_load(open('../config.yaml'))
        self.embedding_client = EmbeddingClient(OpenAI(api_key=self.config['openai_api_key']))
        self.client = OpenAI(api_key=self.config['openai_api_key'])
        
        if subject == "astroPH":
            self.inputs = ['astronomer', 'astronomy']
        else:
            self.inputs = ['machine learning researcher', 'machine learning']
        
    def get_search_results(self, embedding):
        sims = np.dot(self.abstract_embeddings, embedding).flatten()
        topk_indices_search = np.argsort(sims)[::-1][:10]
        
        result_str = ""
        for i in topk_indices_search:
            result_str += self.abstract_texts['abstracts'][i] + '\n\n'
        
        return result_str

    def compare_retrieval(self, query, model, mode = "individual"):
        query_emb = torch.tensor(self.embedding_client.embed(query)).to(device)
        query_act = model.ae(query_emb)
        query_search_results = self.get_search_results(query_emb)

        choice_label, down_clamp_search_results, down_rephrase_search_results = self.test_downweight(query, query_act, model, mode = mode)
        orthogonal_label, up_clamp_search_results, up_rephrase_search_results = self.test_upweight(query, query_act, model, mode = mode)

        up_results = self.score_comparison(orthogonal_label, query_search_results, up_clamp_search_results, up_rephrase_search_results, mode = 'up')
        down_results = self.score_comparison(choice_label, query_search_results, down_clamp_search_results, down_rephrase_search_results, mode = 'down')

        return {'up:': up_results, 'down': down_results} 
    
    def guess_retrieval(self, query, model):
        query_emb = torch.tensor(self.embedding_client.embed(query)).to(device)
        query_act = model.ae(query_emb)
        query_search_results = self.get_search_results(query_emb)

        choice_label, all_label_names, down_clamp_search_results = self.test_downweight(query, query_act, model, no_rephrase = True)

        mult_choice = ""
        for i, label in enumerate(all_label_names):
            mult_choice += str(i) + ": " + label + "\n"
        
        guess_prompt = """You are an expert {} evaluating a retrieval system for research papers.
        
                        INPUT DESCRIPTION:
                        You will be given two sets of titles and abstracts corresponding to research papers retrieved for a query. 
                        You will also be given 5 potential topics, numbered 1 to 5. The importance of exactly one of these topics has been upweighted between the first and second set of papers.
                        
                        OUTPUT DESCRIPTION:
                        Write out a list of specific {} topics covered by the first set, and another list of topics covered by the second set.
                        Compare and contrast the two sets of search results. Based on the 5 potential topics, determine which topic has most likely been upweighted in the second set.
                        If multiple topics are equally likely to have been upweighted, return the most specific topic.
                        Return your answer in the format FINAL: <number>

                        ORIGINAL SEARCH RESULTS: {}
                        --------------------------------
                        MODIFIED SEARCH RESULTS: {}
                        --------------------------------
                        POTENTIAL TOPICS: {}
                        """.format(self.inputs[0], self.inputs[1], query_search_results, down_clamp_search_results, mult_choice)

        guess, text = llm_prompt(guess_prompt, self.client, return_text = True, model = 'gpt-4o-mini')

        return {'label': choice_label, 'choices': all_label_names, 'guess': guess, 'text': text}


    def score_comparison(self, label, original_results, clamp_results, rephrase_results, mode = "up"):
        if mode == "up": words = ['upweights', 'upweighting']
        elif mode == "down": words = ['downweights', 'downweighting']
        comparison_prompt = """You are an expert {} evaluating a retrieval system for research papers. 
                                    
                            INPUT DESCRIPTION:
                            You will be given three sets of titles and abstracts corresponding research papers retrieved for a query.
                            The first set are the papers originally retrieved, with no weighting. The second set are the papers retrieved by Method 1. The third set are the papers retrieved by Method 2.
                            
                            OUTPUT DESCRIPTION:
                            The goal of Method 1 and Method 2 is to {} the following topic in the search results: {}.
                            
                            Evaluate how well each method {} the target topic compared to the original search results.
                            Give each method a score from 1 to 10, 1 meaning no meaningful change from the original search results and 10 being a high-fidelity {} of the choice topic. 
                            Return your scores in the format FINAL: (<Method 1 score>, <Method 2 score>)
                            
                            Here are the search results (paper titles and abstracts):
                            ORIGINAL SEARCH RESULTS: {}
                            --------------------------------
                            METHOD 1 SEARCH RESULTS: {}
                            --------------------------------
                            METHOD 2 SEARCH RESULTS: {}""".format(self.inputs[0], words[0], label, words[0], words[1], original_results, clamp_results, rephrase_results)
        
        results, response_text = llm_prompt(comparison_prompt, self.client, return_text = True, model = 'gpt-4o-mini')
        return results, response_text

    def test_downweight(self, query, query_act, model, klim = 5, no_rephrase = False, mode = "individual"):
        # no rephrase --> use for interpretability

        query_topk = query_act[1]['topk_indices'].numpy()
        query_latents = query_act[1]['latents_post_act'].detach().clone()

        clean_ids = model.clean_labels_by_id.keys()
        query_topk = [ind for ind in query_topk if ind in clean_ids] # only the highly interpretable ones
        top_k_subset = query_topk[:klim]
        choice_k = np.random.choice(top_k_subset, 1)[0]
        choice_label = model.clean_labels_by_id[choice_k]['label']
        
        if no_rephrase: multiplier = 5 # for self-consistency evals --> upweight topic from the top k
        else: multiplier = 0
        
        query_latents[choice_k] *= multiplier
        down_clamped_emb = model.ae.decoder(query_latents) + model.ae.pre_bias
        down_clamp_search_results = self.get_search_results(down_clamped_emb)

        if no_rephrase:
            return choice_label, [model.clean_labels_by_id[ind]['label'] for ind in top_k_subset], down_clamp_search_results

        down_rephrase_prompt = """You are an expert {} trying to construct a query to answer your research questions. 
                                    Given the following query: {}, rephrase it so that it downweights the importance of this specific topic: {}
                                    The goal of the query should be to decrease the impact of this topic on the search results.
                                    Return your answer in the format FINAL: <rewritten query>""".format(self.inputs[0], query, choice_label)
        
        down_rephrase = llm_prompt(down_rephrase_prompt, self.client)
        down_rephrase_emb = self.embedding_client.embed(down_rephrase)
        down_rephrase_search_results = self.get_search_results(down_rephrase_emb)

        return choice_label, down_clamp_search_results, down_rephrase_search_results

    def test_upweight(self, query, query_act, model, mode = "individual"):
        orthogonal_topic_prompt = """You are an expert {} trying to help answer expert-level questions based on research papers.
                    
                    INPUT DESCRIPTION:
                    You will be given a potential query that can be answered with research-level {} literature.

                    OUTPUT DESCRIPTION:
                    List out all topics covered by the query, ranging from the most abstract to the most general.
                    Then given your knowledge of {} research, think of an {}-related topic that is not currently covered by the query, but that might still be relevant to the answer. This can be more abstract or more specific.
                    Summarize this topic in 1-8 words and return it in the format FINAL: <topic>
                    
                    Here is the query: {}""".format(self.inputs[0], self.inputs[1], self.inputs[1], self.inputs[1], query)

        orthogonal_topic = llm_prompt(orthogonal_topic_prompt, self.client)
        orthogonal_emb = self.embedding_client.embed(orthogonal_topic)
        feature_match = np.dot(model.feature_vectors, orthogonal_emb).flatten()

        for index in np.argsort(feature_match)[::-1]:
            if index in model.clean_labels_by_id.keys():
                orthogonal_ind = index
                break
        
        orthogonal_label = model.clean_labels_by_id[orthogonal_ind]['label']

        query_latents = query_act[1]['latents_post_act'].detach().clone()
        query_latents[orthogonal_ind] = 1
        up_clamped_emb = model.ae.decoder(query_latents) + model.ae.pre_bias
        up_clamp_search_results = self.get_search_results(up_clamped_emb)

        print(orthogonal_topic)
        up_rephrase_prompt = """You are an expert {} trying to construct a query to answer your research questions. 
                                    Given the following query: {}, rephrase it so that it upweights the importance of this specific topic: {}
                                    The goal of the query should be to increase the impact of this topic on the search results.
                                    Return your answer in the format FINAL: <rewritten query>""".format(self.inputs[0], query, orthogonal_topic)
        
        up_rephrase = llm_prompt(up_rephrase_prompt, self.client)
        up_rephrase_emb = self.embedding_client.embed(up_rephrase)
        up_rephrase_search_results = self.get_search_results(up_rephrase_emb)

        return orthogonal_label, up_clamp_search_results, up_rephrase_search_results

def save_results(results: List[Dict], filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    subject = "astroPH"
    n_dirs = 9216
    k = 64
    CLAMP_EVAL_RESULTS = 'clamp_eval_results_{}_{}_{}.json'.format(subject, n_dirs, k)
    CAUSAL_EVAL_RESULTS = 'causal_eval_results.json'
    N = 50
    SAE_DATA_DIR = '../saerch/sae_data_{}/'.format('astroPH')
    SAVE_INTERVAL = 5
    emb_path = '../data/vector_store_{}/abstract_embeddings.npy'.format(subject)

    evaluator = Evaluator(subject = subject, embeddings_path = emb_path)

    # load pre-trained model
    model = family.model(sae_data_dir = SAE_DATA_DIR, model_path = '../models/64_9216_128_auxk_astroPH_epoch_50.pth', topk_indices = 'topk_indices_64.npy',
                        topk_values = 'topk_values_64.npy', autointerp_results = 'feature_analysis_results_64.json',
                        mat = 'unnorm_cooccurrence_64.npy', norms = 'occurrence_norms_64.npy', actsims = 'actsims_64.npy')


    real_data = pd.read_csv('../clean_data.csv')
    real_queries = real_data['full_user_query'].dropna()
    cleaned_queries = real_queries.str.replace(r'<@U0.*?>', '', regex=True)
    cleaned_queries = cleaned_queries.unique()
    cleaned_queries = np.random.choice(cleaned_queries, N)
    
    clamp_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {executor.submit(evaluator.compare_retrieval, cleaned_queries[i], model): i 
                           for i in range(len(cleaned_queries))}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                           total=len(cleaned_queries), 
                           desc="Evaluating query clamping"):
            feature_index = future_to_index[future]
            try:
                feature = future.result()
                clamp_results.append(asdict(feature))
                
                if len(clamp_results) % SAVE_INTERVAL == 0:
                    save_results(clamp_results, CLAMP_EVAL_RESULTS)
                    print(f"Checkpoint saved. Processed {len(clamp_results)} features.")
                
            except Exception as exc:
                print(f"Feature {feature_index} generated an exception: {exc}")

    save_results(clamp_results, CLAMP_EVAL_RESULTS)
    print(f"Analysis complete. Results saved to {CLAMP_EVAL_RESULTS}")

    causal_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {executor.submit(evaluator.guess_retrieval, cleaned_queries[i], model): i 
                           for i in range(len(cleaned_queries))}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                           total=len(cleaned_queries), 
                           desc="Evaluating clamping causality"):
            feature_index = future_to_index[future]
            try:
                feature = future.result()
                causal_results.append(asdict(feature))
                
                # Save checkpoint
                if len(causal_results) % SAVE_INTERVAL == 0:
                    save_results(causal_results, CAUSAL_EVAL_RESULTS)
                    print(f"Checkpoint saved. Processed {len(clamp_results)} features.")
                
            except Exception as exc:
                print(f"Feature {feature_index} generated an exception: {exc}")

    save_results(causal_results, CAUSAL_EVAL_RESULTS)
    print(f"Analysis complete. Results saved to {CAUSAL_EVAL_RESULTS}")


if __name__ == "__main__":
    main()
    