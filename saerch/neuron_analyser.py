import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np
import yaml
from openai import AzureOpenAI, OpenAI
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from tqdm import tqdm
import tenacity
from torch.utils.data import DataLoader, TensorDataset
import family
import torch


# Model hyperparameters
CSLG = True
k = 64
ndir = 9216

# Paths
CONFIG_PATH = Path("../config.yaml")
DATA_DIR = Path("../data")
SAE_DATA_DIR = Path("../saerch/sae_data_csLG") if CSLG else Path("../saerch/sae_data_astroPH")
OUTPUT_FILE = Path(f"feature_analysis_results_{k}_{ndir}.json")
# Join OUTPUT_FILE path with SAE_DATA_DIR
OUTPUT_FILE = SAE_DATA_DIR / OUTPUT_FILE
SAVE_INTERVAL = 10

@dataclass
class Feature:
    index: int
    label: str
    reasoning: str
    f1: float
    pearson_correlation: float
    density: float

class BatchNeuronAnalyzer:
    AUTOINTERP_PROMPT = """ 
You are a meticulous {prompt_terms[0]} researcher conducting an important investigation into a certain neuron in a language model trained on {prompt_terms[1]} papers. Your task is to figure out what sort of behaviour this neuron is responsible for -- namely, on what general concepts, features, themes, methodologies or topics does this neuron fire? Here's how you'll complete the task:

INPUT_DESCRIPTION: 

You will be given two inputs: 1) Max Activating Examples and 2) Zero Activating Examples.

- MAX_ACTIVATING_EXAMPLES_DESCRIPTION
You will be given several examples of text that activate the neuron, along with a number being how much it was activated. This means there is some feature, theme, methodology, topic or concept in this text that 'excites' this neuron.

You will also be given several examples of text that don't activate the neuron. This means the feature, topic or concept is not present in these texts.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks.

Step 1: Based on the MAX_ACTIVATING_EXAMPLES provided, write down potential topics, concepts, themes, methodologies and features that they share in common. These will need to be specific - remember, all of the text comes from {prompt_terms[1]}, so these need to be highly specific {prompt_terms[1]} concepts. You may need to look at different levels of granularity (i.e. subsets of a more general topic). List as many as you can think of. Give higher weight to concepts more present/prominent in examples with higher activations.
Step 2: Based on the zero activating examples, rule out any of the topics/concepts/features listed above that are in the zero-activating examples. Systematically go through your list above.
Step 3: Based on the above two steps, perform a thorough analysis of which feature, concept or topic, at what level of granularity, is likely to activate this neuron. Use Occam's razor, as long as it fits the provided evidence. Be highly rational and analytical here.
Step 4: Based on step 4, summarise this concept in 1-8 words, in the form "FINAL: <explanation>". Do NOT return anything after these 1-8 words.

Here are the max-activating examples:

{max_activating_examples}

Here are the zero-activating examples:

{zero_activating_examples}

Work through the steps thoroughly and analytically to interpret our neuron.
"""

    PREDICTION_BASE_PROMPT = """
You are a {prompt_terms[0]} expert that is predicting which abstracts will activate a certain neuron in a language model trained on {prompt_terms[1]} papers. 
Your task is to predict which of the following abstracts will activate the neuron the most. Here's how you'll complete the task:

INPUT_DESCRIPTION:
You will be given the description of the type of paper abstracts on which the neuron activates. This description will be short.

You will then be given an abstract. Based on the concept of the abstract, you will predict whether the neuron will activate or not.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks.

Step 1: Based on the description of the type of paper abstracts on which the neuron activates, reason step by step about whether the neuron will activate on this abstract or not. Be highly rational and analytical here. The abstract may not be clear cut - it may contain topics/concepts close to the neuron description, but not exact. In this case, reason thoroughly and use your best judgement. However, do not speculate on topics that are not present in the abstract.
Step 2: Based on the above step, predict whether the neuron will activate on this abstract or not. If you predict it will activate, give a confidence score from 0 to 1 (i.e. 1 if you're certain it will activate because it contains topics/concepts that match the description exactly, 0 if you're highly uncertain). If you predict it will not activate, give a confidence score from -1 to 0.
Step 3: Provide the final confidence score in the form "PREDICTION: (your prediction)" e.g. "PREDICTION: 0.5". Do NOT return anything after this.

Here is the description/interpretation of the type of paper abstracts on which the neuron activates:
{description}

Here is the abstract to predict:
{abstract}

Work through the steps thoroughly and analytically to predict whether the neuron will activate on this abstract.
"""

    FAMILY_PREDICTION_BASE_PROMPT = """
You are a {prompt_terms[0]} expert that is predicting which abstracts will activate a certain neuron in a language model trained on {prompt_terms[1]} papers. 
Your task is to predict whether the following abstract will activate the neuron the most. Here's how you'll complete the task:

INPUT_DESCRIPTION:
You will be given the description of the type of paper abstracts on which the neuron activates. This description will be short. The neuron is highly general and fires on a family of related topics.

You will then be given an abstract. Based on the concept of the abstract, you will predict whether the neuron will activate or not.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks.

Step 1: Based on the description of the type of paper abstracts on which the neuron activates, reason step by step about whether the neuron will activate on this abstract or not. Be highly rational and analytical here. Reason thoroughly and use your best judgement. The abstract may not exactly match the words or concepts in the description, but it should cover highly related topics. However, do not speculate on topics that are not covered by the abstract.
Step 2: Based on the above step, predict whether the neuron will activate on this abstract or not. If you predict it will activate, give a confidence score from 0 to 1 (i.e. 1 if you're certain it will activate because it contains topics/concepts that match the description exactly, 0 if you're highly uncertain). If you predict it will not activate, give a confidence score from -1 to 0.
Step 3: Provide the final confidence score in the form "PREDICTION: (your prediction)" e.g. "PREDICTION: 0.5". Do NOT return anything after this.

Here is the description/interpretation of the type of paper abstracts on which the neuron activates:
{description}

Here is the abstract to predict:
{abstract}

Work through the steps thoroughly and analytically to predict whether the neuron will activate on this abstract.
"""

    def __init__(self, config_path: Path):

        if CSLG:
            self.PROMPT_TERMS = ['computer science', 'computer science']
        else:
            self.PROMPT_TERMS = ['AI and astronomy', 'astronomy']
        self.config = self.load_config(config_path)
        self.azure_client = AzureOpenAI(
            azure_endpoint=self.config["base_url"],
            api_key=self.config["azure_api_key"],
            api_version=self.config["api_version"],
        )
        self.client = OpenAI(api_key=self.config['jwu_openai_key']) #api_key=self.config['openai_api_key'])
        self.topk_indices, self.topk_values = self.load_sae_data()
        self.abstract_texts = self.load_abstract_texts()
        self.embeddings = self.load_embeddings()

    @staticmethod
    def load_config(config_path: Path) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_sae_data(self) -> Tuple[np.ndarray, np.ndarray]:
        topk_indices = np.load(SAE_DATA_DIR / f"topk_indices_{k}_{ndir}.npy")
        topk_values = np.load(SAE_DATA_DIR / f"topk_values_{k}_{ndir}.npy")
        print(f"Loaded SAE data for k={k} from {SAE_DATA_DIR}")
        return topk_indices, topk_values

    def load_abstract_texts(self) -> Dict:
        if CSLG:
            with open(DATA_DIR / "vector_store_csLG/abstract_texts.json", 'r') as f:
                return json.load(f)

        else:
            with open(DATA_DIR / "vector_store_astroPH/abstract_texts.json", 'r') as f:
                return json.load(f)

    def load_embeddings(self) -> np.ndarray:
        if CSLG:
            return np.load(DATA_DIR / "vector_store_csLG/abstract_embeddings.npy")
        else:
            return np.load(DATA_DIR / "vector_store_astroPH/abstract_embeddings.npy")
        

    def get_feature_activations(self, feature_index: int, m: int, min_length: int = 100, sample = True) -> Tuple[List[Tuple], List[Tuple]]:
        doc_ids = self.abstract_texts['doc_ids']
        abstracts = self.abstract_texts['abstracts']
        
        if isinstance(feature_index, list): # feature family, sample from top bins
            if sample:
                bins_dict = {}
                for bin_num in range(1, 11):
                    bins_dict[bin_num] = []

                for i in feature_index:
                    feature_mask = self.topk_indices == i
                    activated_indices = np.where(feature_mask.any(axis=1))[0]
                    all_activation_values = np.where(feature_mask, self.topk_values, 0).max(axis=1)
                    activation_values = all_activation_values[activated_indices]
                    # Calculate the percentiles
                    percentiles = np.percentile(activation_values, np.arange(0, 101, 10))

                    # Assign the indices to the bins
                    bins = np.digitize(activation_values, percentiles)
                    for bin_num in range(1, 11):
                        bin_indices = activated_indices[bins == bin_num]
                        bins_dict[bin_num] += (bin_indices.tolist())

                to_sample_from = [bins_dict[bin_num] for bin_num in range(10, 11)]
                to_sample_from = [item for sublist in to_sample_from for item in sublist]

                top_m_abstracts = []
                top_m_indices = []
                while len(top_m_indices) < m:
                    i = random.choice(to_sample_from)
                    if len(abstracts[i]) > min_length and i not in top_m_indices:
                        top_m_abstracts.append((doc_ids[i], abstracts[i], all_activation_values[i]))
                        top_m_indices.append(i)
            else: # sample from top values across all feature family members
                feature_mask = self.topk_indices == feature_index
                activated_indices = np.where(feature_mask.any(axis=1))[0]
                activation_values = np.where(feature_mask, self.topk_values, 0).max(axis=1)

                sorted_activated_indices = activated_indices[np.argsort(-activation_values[activated_indices])]

                top_m_abstracts = []
                top_m_indices = []
                for i in sorted_activated_indices:
                    if len(abstracts[i]) > min_length:
                        top_m_abstracts.append((doc_ids[i], abstracts[i], activation_values[i]))
                        top_m_indices.append(i)
                    if len(top_m_abstracts) == m:
                        break

        else: # single feature, use mask and take top values
            feature_mask = self.topk_indices == feature_index
            feature_mask = np.isin(self.topk_indices, feature_index)
            activated_indices = np.where(feature_mask.any(axis=1))[0]
            activation_values = np.where(feature_mask, self.topk_values, 0).max(axis=1)
            sorted_activated_indices = activated_indices[np.argsort(-activation_values[activated_indices])]

            top_m_abstracts = []
            top_m_indices = []
            for i in sorted_activated_indices:
                if len(abstracts[i]) > min_length:
                    top_m_abstracts.append((doc_ids[i], abstracts[i], activation_values[i]))
                    top_m_indices.append(i)
                if len(top_m_abstracts) == m:
                    break

        # Find abstracts where the feature has zero activation
        zero_activation_indices = np.where(~feature_mask.any(axis=1))[0]

        # Randomly sample m abstracts with zero activation and length > min_length
        zero_activation_samples = []
        random.shuffle(list(zero_activation_indices))
        for i in zero_activation_indices:
            if len(abstracts[i]) > min_length:
                zero_activation_samples.append((doc_ids[i], abstracts[i], 0))
            if len(zero_activation_samples) == m:
                break

        return top_m_abstracts, zero_activation_samples

    def generate_family_feature(self, parent, feature_names):
        feature_str = "\n".join(feature_names)
        if CSLG:
            terms = ["machine learning", "artificial intelligence"]
        else:
            terms = ["astronomy", "astrophysics"]
        
        prompt = """You are an expert researcher trying to understand a group of scientific concepts related to {} and {}.
                    
                    INPUT_DESCRIPTION: 
                    You will be given a list of specific {} and {} concepts. There will be 1 parent feature and many child features.
                    
                    OUTPUT_DESCRIPTION:
                    Given the inputs provided, complete the following tasks. Think carefully and rationally, and draw upon your understanding of advanced scientific concepts. 
                    
                    STEP 1: Think about what general concept these specific concepts are related to and list out possible ideas. The general concept may be very similar to the parent concept, but not necessarily.
                    STEP 2: Choose the high-level concept that best summarizes the specific concepts. Be as specific as possible, but make sure the general concept encompasses the majority of the specific concepts.
                    STEP 3: Make sure you do not include "in {}" or "in {}" in your final result. Provide the concept in 2 to 8 words in the form "FINAL: <concept>". Do NOT return anything after this.
                    
                    Here are the specific concepts: 
                    Parent feature: {}
                    ---------------
                    Child features: {}
                    """.format(terms[0], terms[1], terms[0], terms[1], terms[0], terms[1], parent, feature_str)
        
        #print(prompt)

        response = self.client.chat.completions.create(
                model="gpt-4o", #"gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        response_text = response.choices[0].message.content
        #print(response_text)
        try:
            prediction = response_text.split("FINAL:")
            if len(prediction) == 2:
                prediction = prediction[1].strip()
            else:
                prediction = prediction[-1].strip()
        except Exception as e:
            prediction = parent
            print(f"{parent} Error: {e}")
        #print(parent)
        return prediction, response_text

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        #retry=tenacity.retry_if_exception_type(Exception)
    )
    def generate_interpretation(self, top_abstracts: List[Tuple], zero_abstracts: List[Tuple]) -> str:
        max_activating_examples = "\n\n------------------------\n".join([f"Activation:{activation:.3f}\n{abstract}" for _, abstract, activation in top_abstracts])
        zero_activating_examples = "\n\n------------------------\n".join([abstract for _, abstract, _ in zero_abstracts])
        
        prompt = self.AUTOINTERP_PROMPT.format(
            prompt_terms = self.PROMPT_TERMS,
            max_activating_examples=max_activating_examples,
            zero_activating_examples=zero_activating_examples
        )
        
        response = self.azure_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        
        return response.choices[0].message.content

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=25),
        stop=tenacity.stop_after_attempt(7),
        #retry=tenacity.retry_if_exception_type(Exception)
    )
    def predict_activation(self, interpretation: str, abstract: str) -> float:
        prompt = self.FAMILY_PREDICTION_BASE_PROMPT.format(prompt_terms = self.PROMPT_TERMS, description=interpretation, abstract=abstract)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        response_text = response.choices[0].message.content
        #print(response_text)
        try:
            prediction = response_text.split("PREDICTION:")[1].strip()
            return float(prediction.replace("*", ""))
        except Exception:
            return 0.0

    def predict_activations(self, interpretation: str, abstracts: List[str]) -> List[float]:
        return [self.predict_activation(interpretation, abstract) for abstract in abstracts]

    @staticmethod
    def evaluate_predictions(ground_truth: List[int], predictions: List[float]) -> Tuple[float, float]:
        correlation, _ = pearsonr(ground_truth, predictions)
        binary_predictions = [1 if p > 0 else 0 for p in predictions]
        f1 = f1_score(ground_truth, binary_predictions)
        return correlation, f1

    def analyze_feature(self, feature_index: int = None, num_samples: int = 10, feature_dict = None, ) -> Feature:
        num_interp_samples = 5
        if feature_index is None: feature_index = feature_dict['index']
        top_abstracts, zero_abstracts = self.get_feature_activations(feature_index, num_samples)
        
        if feature_dict is None:
            interpretation_full = self.generate_interpretation(top_abstracts[:num_interp_samples], zero_abstracts[:num_interp_samples])
            interpretation = interpretation_full.split("FINAL:")[1].strip().replace("*", "")
        else:
            interpretation = feature_dict['label']
            interpretation_full = feature_dict['reasoning']

        num_test_samples = 3
        test_abstracts = [abstract for _, abstract, _ in top_abstracts[num_interp_samples: num_interp_samples + num_test_samples] + zero_abstracts[num_interp_samples: num_interp_samples + num_test_samples]]
        ground_truth = [1] * len(top_abstracts[num_interp_samples: num_interp_samples + num_test_samples]) + [0] * len(zero_abstracts[num_interp_samples: num_interp_samples + num_test_samples])
        predictions = self.predict_activations(interpretation, test_abstracts)
        print(f"Predictions: {predictions}, ground truth: {ground_truth}")
        correlation, f1 = self.evaluate_predictions(ground_truth, predictions)

        density = (self.topk_indices == feature_index).any(axis=1).mean()

        return Feature(
            index=feature_index,
            label=interpretation,
            reasoning=interpretation_full,
            f1=f1,
            pearson_correlation=correlation,
            density=density
        )
    
    def interp_superfeature(self, family, model):
        all_names, ids = family.get_names_and_ids(model)
        superfeature, superfeature_response = self.generate_family_feature(all_names[-1], all_names[:-1])
        #print(superfeature)
        
        top_abstracts, zero_abstracts = self.get_feature_activations(feature_index = ids, m = 5)

        num_test_samples = 4
        test_abstracts = [abstract for _, abstract, _ in top_abstracts[:num_test_samples] + zero_abstracts[:num_test_samples]]
        ground_truth = [1] * num_test_samples + [0] * num_test_samples

        predictions = self.predict_activations(superfeature, test_abstracts)
        correlation, f1 = self.evaluate_predictions(ground_truth, predictions)

        print(f"Pearson correlation: {correlation}")
        print(f"F1 score: {f1}")

        feature_pearsons = [model.clean_labels[feature]['pearson_correlation'] for feature in all_names]
        feature_f1s = [model.clean_labels[feature]['f1'] for feature in all_names]
        
        return {'predictions': predictions, 'superfeature': superfeature, 'super_reasoning': superfeature_response, 'family_f1': f1, 'family_pearson': correlation, 'feature_f1': feature_f1s, 'feature_pearson': feature_pearsons, 'feature_names': all_names}

def save_results(results: List[Dict], filename: Path):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filename: Path) -> List[Dict]:
    if filename.exists():
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def analyze_all_features(analyzer, num_samples = 10, save_file = None):
    # for individual features
    if save_file is None: save_file = OUTPUT_FILE # use default output file

    results = load_results(save_file) 
    existing_indices = [feature['index'] for feature in results]
    all_indices = list(range(ndir))
    missing = [i for i in all_indices if i not in existing_indices]
    print("Starting analysis from index {}".format(min(missing)))

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {
            executor.submit(analyzer.analyze_feature, i, num_samples): i
            for i in missing
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(missing),
            desc="Analysing features"
        ):
            feature_index = future_to_index[future]
            try:
                feature = future.result()
                results.append(asdict(feature))

                # Save checkpoint
                if len(results) % SAVE_INTERVAL == 0:
                    save_results(results, OUTPUT_FILE)
                    print(f"Checkpoint saved. Processed {len(results)} features.")

            except Exception as exc:
                print(f"Feature {feature_index} generated an exception: {exc}")

    save_results(results, OUTPUT_FILE)
    print(f"Analysis complete. Results saved to {OUTPUT_FILE}")


def analyze_family(analyzer, families, model, save_file):
    # runs interp_superfeature on each 
    start_index = 0
    print(f"Starting analysis from family {start_index}...")

    results = load_results(save_file)
    completed = [result['index'] for result in results]
    family_id_list = list(families.keys())

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {
            executor.submit(analyzer.interp_superfeature, families[family_id_list[i]], model): i
            for i in family_id_list if i not in completed
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(family_id_list) - start_index,
            desc="Analysing families"
        ):
            feature_index = future_to_index[future]
            try:
                feature = future.result()
                feature['index'] = family_id_list[feature_index]
                results.append(feature)

                # Save checkpoint
                if len(results) % SAVE_INTERVAL == 0:
                    save_results(results, save_file)
                    print(f"Checkpoint saved. Processed {len(results)} families.")

            except Exception as exc:
                print(f"Family {feature_index} generated an exception: {exc}")

    save_results(results, save_file)
    print(f"Analysis complete. Results saved to {save_file}")


def main():
    analyzer = BatchNeuronAnalyzer(CONFIG_PATH)  # Presumably you have this class defined
    mode = "individual"  # Change this to 'individual', 'rerun', or 'family' as needed
    num_samples = 10

    if mode == "individual":
        analyze_all_features(analyzer, num_samples)


if __name__ == "__main__":
    main()
