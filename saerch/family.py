import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
import matrix
import pandas as pd
from topk_sae import FastAutoencoder, loss_fn, unit_norm_decoder_grad_adjustment_, unit_norm_decoder_, init_from_data_
from autointerp import NeuronAnalyzer
import networkx as nx
import os
import re

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024
CONFIG = '../config.yaml'

def nn_hierarchy(ae_large, ae_small, save_path = None): # nearest neighbors hierarchy
    feats = np.arange(ae_small.n_dirs)
    nns = {int(feat): [] for feat in feats}

    # nearest neighbors of features (decoder weights)
    sims = np.dot(ae_small.decoder.weight.data.cpu().numpy().T, ae_large.decoder.weight.data.cpu().numpy()) # 3072 x 9216
    matches = np.argmax(sims, axis = 0)
    
    #print(matches.shape, np.max(matches), np.min(matches))

    for i, match in enumerate(matches):
        if match in nns.keys():
            nns[int(match)].append(i)
    
    if save_path is not None:
        json.dump(nns, open(save_path, 'w'))

    return nns

def clean(label):
    label = label.split(' in astronomy')[0].split(' in astrophysics')[0].split(' in physics')[0].split(' in astronomical')[0].split(' in astrophysical')[0]
    label = label.split(' in Astronomy')[0].split(' in Astrophysics')[0].split(' in Physics')[0].split(' in Astronomical')[0].split(' in Astrophysical')[0]
    label = label.split(' in computer science')[0].split(' in CS')[0]
    if len(re.findall(r"""["'][A-Za-z]["']""", label)) > 0:
        return ""
    if len(re.findall(r"[Kk]eyword", label)) > 0:
        return ""
    if len(re.findall(r"[Tt]opic", label)) > 0:
        return ""
    if len(re.findall(r"[Cc]oncept", label)) > 0:
        return ""
    if len(re.findall(r"[Pp]resence", label)) > 0:
        return ""
    return label

def get_clean_labels(auto_results, density_threshold = 1):
    clean_labels = {}
    for result in auto_results:
        label = result['label']
        clean_label = clean(label)
        
        if clean_label != "":
            result['clean_label'] = clean_label
            if result['f1'] >= 0.8 and result['pearson_correlation'] >= 0.8 and result['density'] < density_threshold:
                if clean_label not in clean_labels.keys():
                    clean_labels[clean_label] = {'index': result['index'], 'density': result['density'], 'f1': result['f1'], 'pearson_correlation': result['pearson_correlation'],}
                else:
                    # keep only the highest-scoring label if repeats
                    existing = clean_labels[clean_label]
                    if result['f1'] + result['pearson_correlation'] > existing['f1'] + existing['pearson_correlation']:
                        clean_labels[clean_label] = {'index': result['index'], 'density': result['density'], 'f1': result['f1'], 'pearson_correlation': result['pearson_correlation'],}
    
    return clean_labels

def get_cosine_sim(weight1, weight2, targets1, targets2):
    A = weight1[targets1] # n1 x 1536
    B = weight2[targets2] # 1536 x n2

    return np.dot(A, B.T)


class FeatureFamily():
    def __init__(self, parent, children):
        self.parent = parent
        self.children = children
    
    def exp_json(self):
        return {'parent': self.parent, 'children': self.children}

    def get_names_and_ids(self, model):
        children = [child[0] for child in self.children]
        ids = [child[1] for child in self.children]
        densities = [model.norms[id] for id in ids]
        all_names = children + [self.parent[0]]
        
        ids = list(np.array(ids)[np.argsort(densities)])
        ids.append(model.clean_labels[self.parent[0]]['index'])

        return all_names, ids

    def matrix(self, model, mode = 'cooc', return_mat = False):
        all_names, ids = self.get_names_and_ids(model)

        if mode == 'cooc':
            min_matrix = np.minimum.outer(model.norms[ids], model.norms[ids])
            norm_mat = model.mat[ids][:, ids]
            norm_mat /= min_matrix
        elif mode == 'actsims':
            norm_mat = model.actsims[ids][:, ids]
        elif mode == 'cosine':
            feature_vecs = model.feature_vectors[ids]
            norm_mat = np.dot(feature_vecs, feature_vecs.T)
        
        last_row = norm_mat[-1, :-1]
        rest_of_matrix = norm_mat[:-1, :-1]

        ratio = np.mean(last_row) / np.mean(rest_of_matrix)
        if return_mat:
            return norm_mat, ratio
        
        return ratio


class MultiModel():
    def __init__(self, model_list, all_onehots, all_acts):
        self.model_list = model_list
        self.index, self.feature_index = self.generate_index() # matrix index to feature

        self.norms = np.hstack(tuple([model.norms[list(model.clean_labels_by_id.keys())] for model in model_list])) # for one-hot co-occurrence
        self.feature_vectors = np.vstack(tuple([model.feature_vectors[list(model.clean_labels_by_id.keys())] for model in model_list]))

        if os.path.exists(all_acts) and os.path.exists(all_onehots):
            self.actsims = np.load(all_acts)
            self.mat = np.load(all_onehots)
        
        else: # acts are already normalized by feature (dim = 0)
            print('Generating matrices...')
            acts = torch.concat(tuple([model.acts[:, list(model.clean_labels_by_id.keys())] for model in model_list]), dim = 1)
            onehots = torch.concat(tuple([model.onehots[:, list(model.clean_labels_by_id.keys())] for model in model_list]), dim = 1)

            self.actsims, norms = matrix.co_occurrence(acts)
            np.save(all_acts, self.actsims)
            
            self.mat, norms = matrix.co_occurrence(onehots)
            np.save(all_onehots, self.mat)
            print('Generated and saved matrices.')
        
        self.cosine_sim = np.dot(self.feature_vectors, self.feature_vectors.T)
    
    def generate_index(self):
        ids = []
        index = []
        feature_index = {}

        tally = 0
        for i, model in enumerate(self.model_list):
            model_ids = model.clean_labels_by_id.keys()
            ids += list()

            for j, id in enumerate(model_ids):
                index.append(model.clean_labels_by_id[id]['label'])
                feature_index[(i, id)] = tally
                tally += 1
        
        return index, feature_index
    

    def explore_splitting(self, ind, nns_list, names = ["SAE16", "SAE32", "SAE64"], verbose = True):
        targets = [ind]
        matrix_indices = [] # initial index
        feature_names = []

        for i, model in enumerate(self.model_list):
            if verbose: print(i, targets, model.get_feature_names(targets))
            
            next_targets = []
            
            j = 0
            set2 = set()
            for target in targets:
                if j < 10:
                    try:
                        matrix_indices.append(self.feature_index[(i, target)])
                        feature_names.append((names[i], j, model.clean_labels_by_id[target]['label']))
                        j += 1
                    except:
                        if verbose: print('{} not in clean feature list'.format(target))
                    
                    if i < len(self.model_list) - 1:
                        matches = nns_list[i][target]
                        if len(matches) > 0:
                            # print(get_cosine_sim(model.feature_vectors, self.model_list[i + 1].feature_vectors, target, matches))
                            next_targets += matches
                    else:
                        set2 = model.get_feature_names(targets)
                
            targets = next_targets
        
        # 16 --> 64 directly
        if len(nns_list) == len(self.model_list):
            if verbose: print('self consistency ', nns_list[-1][ind], self.model_list[-1].get_feature_names(nns_list[-1][ind]))
            if verbose: print(get_cosine_sim(self.model_list[0].feature_vectors, self.model_list[-1].feature_vectors, ind, nns_list[-1][ind]))

        set1 = self.model_list[-1].get_feature_names(nns_list[-1][ind])
        if len(set(set1).union(set(set2))) == 0:
            intersection = 0
        else:
            intersection = len(set(set1).intersection(set(set2))) / len(set(set1).union(set(set2)))
        return matrix_indices, feature_names, intersection
        

class Model():
    def __init__(self, sae_data_dir, model_path, autointerp_results, dataloader, num_abstracts, topk_indices = None, topk_values = None, 
                 mat = None, norms = None, actsims = None, d_model = 1536):
        
        self.k, self.n_dirs, auxk = model_path.split('/')[-1].split('_')[:3]
        self.k = int(self.k)
        self.n_dirs = int(self.n_dirs)
        self.sae_data_dir = sae_data_dir
        self.dataloader = dataloader
        self.num_abstracts = num_abstracts

        mat = sae_data_dir + 'unnorm_cooccurrence_{}_{}.npy'.format(self.k, self.n_dirs) if mat is None else mat
        norms = sae_data_dir + 'norms_{}_{}.npy'.format(self.k, self.n_dirs) if norms is None else norms
        actsims = sae_data_dir + 'actsims_{}_{}.npy'.format(self.k, self.n_dirs) if actsims is None else actsims
        topk_indices = sae_data_dir + 'topk_indices_{}_{}.npy'.format(self.k, self.n_dirs) if topk_indices is None else topk_indices
        topk_values = sae_data_dir + 'topk_values_{}_{}.npy'.format(self.k, self.n_dirs) if topk_values is None else topk_values
        
        autointerp_results = sae_data_dir + autointerp_results

        self.ae = FastAutoencoder(n_dirs = self.n_dirs, k = self.k, d_model = d_model, auxk = int(auxk), multik = 0)
        self.ae.load_state_dict(torch.load(model_path))
        self.ae.eval()
        self.feature_vectors = self.ae.decoder.weight.data.cpu().numpy().T

        
        if os.path.exists(topk_indices) and os.path.exists(topk_values):
            self.topk_indices = np.load(topk_indices)
            self.topk_values = np.load(topk_values)
        else:
            self.topk_indices, self.topk_values = self.generate_topk(topk_indices, topk_values)

        self.auto_results = json.load(open(autointerp_results)) 
        self.clean_labels = get_clean_labels(self.auto_results, density_threshold = 0.2) # CHANGE THIS IF YOU WANT
        
        for label in self.clean_labels:
            self.clean_labels[label]['label'] = label
        
        
        self.auto_results_by_id = {label['index']: label for label in self.auto_results}
        self.clean_labels_by_id = {label['index']: label for label in self.clean_labels.values()}
        
        # one-hot co-occurrences
        if os.path.exists(mat) and os.path.exists(norms):
            self.mat = np.load(mat)
            self.norms = np.load(norms)
            self.onehots = matrix.activations(self.topk_indices, self.topk_values, ndir = self.n_dirs, nex = self.num_abstracts, mode = 'onehot')
        else:
            self.onehots, self.mat, self.norms = self.generate_matrix(mat, norms)
        
        # activation dot product
        if os.path.exists(actsims):
            self.actsims = np.load(actsims)
            self.acts = matrix.activations(self.topk_indices, self.topk_values, ndir = self.n_dirs, nex = self.num_abstracts, mode = 'value')
        else:
            self.acts, self.actsims = self.generate_activation_sims(actsims)

        self.clean_cooc = matrix.get_norm_cooc(self.mat, self.norms, self.clean_labels, threshold = 0.1, poisson = False)
        np.fill_diagonal(self.clean_cooc, 0)
        self.neuron = None #NeuronAnalyzer(CONFIG, 1, 10, k = self.k, ndirs = self.n_dirs)

    def generate_topk(self, topk_indices_path = None, topk_values_path = None):
        print('Generating topk indices and values...')
        topk_indices = np.zeros((self.num_abstracts, self.k), dtype=np.int64)
        topk_values = np.zeros((self.num_abstracts, self.k), dtype=np.float32)

        # Process batches
        with torch.no_grad():
            for i, (batch,) in enumerate(tqdm(self.dataloader, desc="Processing abstracts")):
                batch = batch.to(device)
                _, info = self.ae(batch)
                
                start_idx = i * BATCH_SIZE
                end_idx = start_idx + batch.size(0)
                
                topk_indices[start_idx:end_idx] = info['topk_indices'].cpu().numpy()
                topk_values[start_idx:end_idx] = info['topk_values'].cpu().numpy()
        
        if topk_indices_path is not None and topk_values_path is not None:
            np.save(topk_indices_path, topk_indices)
            np.save(topk_values_path, topk_values)

            print("Processing complete. Results saved to {}.".format(topk_indices_path))  

        return topk_indices, topk_values

    def generate_matrix(self, mat_path = None, norms_path = None): # if it doesn't exist
        print('Generating matrix and norms...')
        activations = matrix.activations(self.topk_indices, self.topk_values, nex = self.num_abstracts, ndir = self.n_dirs, mode = 'onehot')
        mat, norms = matrix.co_occurrence(activations)

        if mat_path is not None:
            np.save(mat_path, mat)
            np.save(norms_path, norms)

        return activations, mat, norms
    
    def generate_activation_sims(self, actsim_path):
        print('Generating activation similarities...')
        activations = matrix.activations(self.topk_indices, self.topk_values, nex = self.num_abstracts, ndir = self.n_dirs, mode = 'value')
        mat, norms = matrix.co_occurrence(activations)

        np.save(actsim_path, mat)

        return activations, mat

    def get_feature_activations(self, n, feature_index = None, feature_name = None): # example abstracts
        if feature_index is None:
            if feature_name is None:
                raise ValueError("Either feature_index or feature_name must be provided")
            feature_index = self.clean_labels[feature_name]['index']

        self.neuron.topk_indices = self.topk_indices
        self.neuron.topk_values = self.topk_values
        
        return self.neuron.get_feature_activations(m = n, feature_index = feature_index)

    def get_families(self, n = 3): # iterative family finding in the MST
        G_tree = matrix.make_MST(self.clean_cooc, self.clean_labels)
        subtrees = matrix.subtree_iterate(self.mat, self.norms, G_tree, self.clean_labels, n = n)
        return subtrees
    
    def deduplicate_families(self, families):
        tally = 0
        clean_families = families.copy()
        for family in families:
            for family2 in families:
                if family != family2:
                    intersection = set(families[family].children).intersection(families[family2].children)
                    union = set(families[family].children).union(families[family2].children)
                    if len(intersection) / len(union) > 0.6:
                        tally += 1
                        try:
                            if len(families[family].children) > len(families[family2].children): clean_families.pop(family2)
                            else: clean_families.pop(family)
                        except:
                            pass
                            
        
        print('Deduplicated {} families'.format(tally))
        return clean_families

    def dedup_and_save_families(self, subtrees, save_path = None):
        if save_path is None:
            save_path = self.sae_data_dir + 'clean_families_{}_{}.json'.format(self.k, self.n_dirs)
        families = {}
        for iteration in subtrees:
            for subtree in iteration:
                superfeature, children = matrix.print_subtree(subtree, self.clean_labels, all_children = [], verbose = False)
                if len(children) > 3:
                    families[superfeature] = FeatureFamily(parent = [superfeature, subtree[2]], children = children)
        
        families = self.deduplicate_families(families)
        
        json_families = {key: value.exp_json() for key, value in families.items()}
        json.dump(json_families, open(save_path, 'w'))
        return families
    
    def load_all_families(self, n = None):
        path = self.sae_data_dir + 'clean_families_{}_{}.json'.format(self.k, self.n_dirs)
        if not os.path.exists(path) or n is not None:
            print('No saved families found. Generating...')
            if n is None: n = 3
            self.families = self.dedup_and_save_families(self.get_families(n), save_path = path)
            return self.families
        else:
            families = json.load(open(path))
            self.families = {key: FeatureFamily(parent = value['parent'], children = value['children']) for key, value in families.items()}
            return self.families
    
    def get_feature_names(self, indices): # get feature names from indices
        # auto results in format similar to feature_analysis_results.json
        k = len(indices)
        feature_list = [""] * k
        for result in self.auto_results:
            if result['index'] in indices:
                feature_list[list(indices).index(result['index'])] = result['label']
        
        return feature_list