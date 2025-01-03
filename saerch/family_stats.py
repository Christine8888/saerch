import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import family

def get_membership_fraction(interp, model, threshold = 0.8 ):
    # compute fraction of features that are in a family
    clean_interp = [fam for fam in interp if fam['family_f1'] >= threshold and fam['family_pearson'] >= threshold] # threshold
    family_index = {key: [] for key in model.clean_labels.keys()}
    
    for fam in clean_interp:
        for feature in fam['feature_names']:
            if feature in family_index.keys():
                family_index[feature].append(fam['superfeature'])
            else:
                print(feature)
    
    total = len(model.clean_labels.keys())
    in_a_family = 0
    for key in family_index.keys():
        if len(family_index[key]) > 0:
            in_a_family += 1
    
    return in_a_family / total

def block_diagonalize(model, interp):
    # order matrix by family then by feature
    index_order = []
    family_lengths = []
    seen = set()

    for fam in interp:
        family_indices = []
        for feature in fam['feature_names']:
            index = model.clean_labels[feature]['index']
            if index not in seen:
                family_indices.append(index)
                seen.add(index)
        
        if family_indices == []:
            continue
        index_order.extend(family_indices)
        
        if len(family_lengths) == 0:
            family_lengths.append(len(family_indices))
        else:
            family_lengths.append(family_lengths[-1] + len(family_indices))
    
    index_order = list(dict.fromkeys(index_order))
    missing = [i for i in model.clean_labels_by_id.keys() if i not in index_order]

    for i in missing:
        index_order.append(i)
        family_lengths.append(family_lengths[-1] + 1)

    return index_order, family_lengths, seen

def compute_block_ratio(matrix, block_lengths, mode = 'ratio'):
    # compute block ratio of matrix
    block_elements = []
    offdiag_elements = []
    for i in range(len(block_lengths) - 1):
        block = matrix[block_lengths[i]:block_lengths[i + 1], block_lengths[i]:block_lengths[i + 1]]
        
        block_elements.extend(block.flatten())
        if mode == 'ratio':
            offdiag_below = matrix[block_lengths[i]:block_lengths[i + 1], :block_lengths[i]]
            offdiag_above = matrix[block_lengths[i]:block_lengths[i + 1], block_lengths[i + 1]:]
            offdiag = np.concatenate((offdiag_below, offdiag_above), axis = 1)
            offdiag_elements.extend(offdiag.flatten())
    
    if mode == 'ratio': return np.mean(block_elements) / np.mean(offdiag_elements)
    elif mode == 'fraction': return np.sum(block_elements) / np.sum(matrix)

def get_child_mean(interp):
    # compute mean f1 and pearson for children of each family
    child_f1s = []
    child_pearsons = []
    
    for fam in interp:
        child_f1s.extend(fam['feature_f1'])
        child_pearsons.extend(fam['feature_pearson'])
    
    return np.mean(child_f1s), np.mean(child_pearsons)

def get_family_stats(clean_families, interp, model, verbose = True): # clean families is the original family json, interp is the autointerp
    # clean & tabular format for feature family stats
    size = [len(clean_families[fam].children) for fam in clean_families]
    if verbose: print('mean family size', np.mean(size))
    
    if verbose: print('f1', np.nanmean([fam['family_f1'] for fam in interp]))
    if verbose: print('pearson', np.nanmean([fam['family_pearson'] for fam in interp]))
    
    cooc_ratio = [clean_families[fam].matrix(model, mode = 'cooc') for fam in clean_families]
    cooc_ratio = [ratio for ratio in cooc_ratio if not np.isinf(ratio)]
    if verbose: print('mean cooc ratio', np.mean(cooc_ratio))

    index_order, block_lengths, seen = block_diagonalize(model, interp)
    mat_base = model.mat.copy()
    mat_base /= np.minimum.outer(model.norms, model.norms)
    mat_perm = mat_base[index_order][:, index_order]
    random_perm = np.random.permutation(mat_base.shape[0])
    if verbose: print('cooc block diagonal', compute_block_ratio(mat_perm, block_lengths))
    if verbose: print('random block diagonal', compute_block_ratio(mat_base[random_perm][:, random_perm], block_lengths))
    
    act_base = model.actsims.copy()
    act_perm = act_base[index_order][:, index_order]
    if verbose: print('actsim block diagonal', compute_block_ratio(act_perm, block_lengths))
    if verbose: print('random block diagonal', compute_block_ratio(act_base[random_perm][:, random_perm], block_lengths))

    if verbose: print('membership fraction', get_membership_fraction(interp, model))
    
    # Print values in the same row
    print('\\hline')
    print(np.mean(size), '&', np.nanmean([fam['family_f1'] for fam in interp]), '&', np.nanmean([fam['family_pearson'] for fam in interp]), '&', np.mean(cooc_ratio), '&', compute_block_ratio(mat_perm, block_lengths) / compute_block_ratio(mat_base, block_lengths), '&', compute_block_ratio(act_perm, block_lengths) / compute_block_ratio(act_base, block_lengths), '&', get_membership_fraction(interp, model), '\\\\')


def main():
    ns = [3072, 6144, 9216]
    ks = [16, 32, 64]
    auxks = [24, 48, 128]
    SAE_DATA_DIR = '../saerch/sae_data_astroPH/'
    
    mode = 'epoch_astroPH_old'
    SAE_DATA_DIR = '../saerch/sae_data_{}/'.format('csLG')
    
    DATA_DIR = '../data/vector_store_astroPH/'
    abstract_embeddings = np.load(DATA_DIR + "/abstract_embeddings.npy")
    abstract_embeddings = abstract_embeddings.astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(abstract_embeddings))
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    num_abstracts = len(abstract_embeddings)
    
    interps = []
    models = []
    clean_families = []
    for n, k, auxk in zip(ns, ks, auxks):
        interps.append(json.load(open(SAE_DATA_DIR + 'family_analysis_{}_{}.json'.format(k, n))))
        model = family.Model(sae_data_dir = SAE_DATA_DIR, model_path = '../models/{}_{}_{}_auxk_{}_50.pth'.format(k, n, auxk, mode),
                                   autointerp_results='feature_analysis_results_{}_4omini.json'.format(k), dataloader = dataloader, num_abstracts = num_abstracts)
        clean_families.append(model.load_all_families())
        models.append(model)
    
    for i, interp in enumerate(interps):
        get_family_stats(clean_families[i], interp, models[i], verbose = False)

