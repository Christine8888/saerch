import autointerp
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import scipy.stats
import seaborn as sns
import torch
import torch.nn as nn
import topk_sae
import umap
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from pathlib import Path
import scipy.stats
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# UTILITY FUNCTIONS FOR LOADING
def create_dataloader(embeddings: np.ndarray, batch_size: int, shuffle: bool = False):
    # create a dataloader from the embeddings
    dataset = TensorDataset(torch.from_numpy(embeddings))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_pretrained_sae(model_path: str, d_model: int, n_dirs: int, k: int, auxk: int, multik: int):
    # load sae model from path
    ae = topk_sae.FastAutoencoder(n_dirs, d_model, k, auxk, multik=multik).to(device)
    ae.load_state_dict(torch.load(model_path, map_location=device))
    ae.eval()
    return ae

def compute_latent_info(ae_model: nn.Module, dataloader: DataLoader, k: int, multik: int, n_dirs: int, device=device):
    # compute topk indices, values and latents pre activation
    num_samples = len(dataloader.dataset)
    topk_indices = np.zeros((num_samples, k), dtype=np.int64)
    topk_values = np.zeros((num_samples, k), dtype=np.float32)
    multik_indices = np.zeros((num_samples, multik), dtype=np.int64)
    multik_values = np.zeros((num_samples, multik), dtype=np.float32)
    latents_pre_act = np.zeros((num_samples, n_dirs), dtype=np.float32)
    start_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing latent info"):
            x = batch[0].to(device)
            _, info = ae_model(x)
            bs = x.size(0)
            end_idx = start_idx + bs
            topk_indices[start_idx:end_idx] = info["topk_indices"].cpu().numpy()
            topk_values[start_idx:end_idx] = info["topk_values"].cpu().numpy()
            multik_indices[start_idx:end_idx] = info["multik_indices"].cpu().numpy()
            multik_values[start_idx:end_idx] = info["multik_values"].cpu().numpy()
            latents_pre_act[start_idx:end_idx] = info["latents_pre_act"].cpu().numpy()
            start_idx = end_idx
    return {
        "topk_indices": topk_indices,
        "topk_values": topk_values,
        "multik_indices": multik_indices,
        "multik_values": multik_values,
        "latents_pre_act": latents_pre_act,
    }

def load_embeddings_and_texts(embeddings_path: str, abstract_texts_path: str, metadata_path: str = None):
    embeddings = np.load(embeddings_path)
    embeddings = embeddings.astype(np.float32)
    with open(abstract_texts_path, 'r') as f:
        abstract_json = json.load(f)
    abstract_texts = abstract_json['abstracts']
    abstract_ids = abstract_json.get('doc_ids', None)
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        metadata = json.load(open(metadata_path, 'r'))
    return embeddings, abstract_texts, metadata, abstract_ids

def compute_only_topk_indices_values(model: topk_sae.FastAutoencoder, embeddings: np.ndarray, batch_size: int = 1024, save: str = None):
    # get topk indices and values from the model
    dataloader = create_dataloader(embeddings, batch_size=batch_size, shuffle=False)
    num_abstracts = len(embeddings)
    k = model.k
    topk_indices = np.zeros((num_abstracts, k), dtype=np.int64)
    topk_values = np.zeros((num_abstracts, k), dtype=np.float32)
    with torch.no_grad():
        for i, (batch,) in enumerate(tqdm(dataloader, desc="Processing abstracts")):
            batch = batch.to(device)
            _, info = model(batch)
            start_idx = i * batch_size
            end_idx = start_idx + batch.size(0)
            topk_indices[start_idx:end_idx] = info['topk_indices'].cpu().numpy()
            topk_values[start_idx:end_idx] = info['topk_values'].cpu().numpy()
    
    if save is not None:
        os.makedirs(save, exist_ok=True)
        np.save(f"{save}/topk_indices.npy", topk_indices)
        np.save(f"{save}/topk_values.npy", topk_values)

    return topk_indices, topk_values


def extract_abstract_embeddings(embeddings_path: str, index_mapping_path: str, documents_path: str, output_abstract_embeddings_path: str, output_abstract_texts_path: str) -> None:
    embeddings = np.load(embeddings_path)
    with open(index_mapping_path, 'rb') as f:
        index_mapping = pickle.load(f)
    with open(documents_path, 'rb') as f:
        documents = pickle.load(f)
    abstract_embeddings = []
    abstract_texts = []
    doc_ids = []
    for doc_id, mappings in index_mapping.items():
        if 'abstract' in mappings:
            abstract_index = mappings['abstract']
            abstract_embeddings.append(embeddings[abstract_index])
            doc = next((d for d in documents if d.id == doc_id), None)
            if doc:
                abstract_texts.append(doc.abstract)
                doc_ids.append(doc_id)
    abstract_embeddings = np.array(abstract_embeddings)
    np.save(output_abstract_embeddings_path, abstract_embeddings)
    with open(output_abstract_texts_path, 'w') as f:
        json.dump({'doc_ids': doc_ids,'abstracts': abstract_texts}, f)

# SUMMARY STATISTICS AND PLOTTING

def compute_reconstruction_mse_over_k(ae_model: nn.Module, batch: torch.Tensor, k_values: List[int]):
    # compute the reconstruction mse over k values
    # can reproduce multitopk results from https://arxiv.org/abs/2406.04093
    _, info = ae_model(batch)
    latents_pre_act = info["latents_pre_act"]
    mse_scores = []
    for kval in k_values:
        recon_at_k = ae_model.decode_at_k(latents_pre_act, kval)
        mse_val = topk_sae.normalized_mse(recon_at_k, batch)
        mse_scores.append(mse_val)
    return np.array(mse_scores)

def plot_latent_histogram(latents_pre_act: np.ndarray, cutoff: int = 500):
    # plot average activations for all latents -- useful for our matryoshka embeddings findings
    sparsesort = np.flip(np.sort(np.abs(latents_pre_act), axis=1), axis=1)
    sparsecut = sparsesort[:, :cutoff]
    sparsemean = np.mean(sparsecut, axis=0)
    plt.figure()
    plt.plot(sparsemean, linewidth=0.5, color="k")
    plt.plot(np.convolve(sparsemean, np.ones(100) / 100, mode="valid"), color="r")
    plt.xlabel("Latent dimension")
    plt.ylabel("Abs. avg activation value")
    plt.show()

def plot_embedding_histogram(embeddings: np.ndarray):
    embmean = np.mean(np.abs(embeddings), axis=0)
    plt.figure()
    plt.plot(embmean, linewidth=0.5, color="k")
    plt.plot(np.convolve(embmean, np.ones(100) / 100, mode="valid"), color="r")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Abs. avg embedding value")
    plt.ylim(0, embmean.max() * 1.2)
    plt.show()

def plot_mse_vs_k(kvals: List[int], mse_scores: np.ndarray, use_log_scale: bool = True):
    # plot the reconstruction mse over k values
    plt.figure()
    plt.plot(kvals, mse_scores, marker="o")
    if use_log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.xlabel("k")
    plt.ylabel("Normalized MSE")
    plt.show()

def run_autointerp_analysis(config_path: Path, feature_index: int, num_samples: int):
    # run autointerp analysis on single feature
    analyzer = autointerp.NeuronAnalyzer(config_path, feature_index, num_samples)
    top_abstracts, zero_abstracts = analyzer.get_feature_activations(num_samples)
    interpretation = analyzer.generate_interpretation(top_abstracts, zero_abstracts)
    print(f"Interpretation for feature {feature_index}: {interpretation}")
    num_test_samples = 4
    test_abstracts = [abstract for _, abstract, _ in top_abstracts[-num_test_samples:] + zero_abstracts[-num_test_samples:]]
    ground_truth = [1] * num_test_samples + [0] * num_test_samples
    predictions = analyzer.predict_activations(interpretation, test_abstracts)
    correlation, f1 = analyzer.evaluate_predictions(ground_truth, predictions)
    print(f"Pearson correlation: {correlation}")
    print(f"F1 score: {f1}")


def compute_set_overlap_and_correlation(info, idx_a: int, idx_b: int, cutoff: int = 100):
    topk_a = info["topk_indices"][idx_a][:cutoff]
    topk_b = info["topk_indices"][idx_b][:cutoff]
    set_a = set(topk_a)
    set_b = set(topk_b)
    overlap = set_a & set_b
    set_overlap_fraction = len(overlap) / float(cutoff)
    if len(overlap) < 2:
        return set_overlap_fraction, 0.0, 0.0
    overlap_list = list(overlap)
    overlap_order_a = [list(topk_a).index(x) for x in overlap_list]
    overlap_order_b = [list(topk_b).index(x) for x in overlap_list]
    pearsonr_result = scipy.stats.pearsonr(overlap_order_a, overlap_order_b)
    pearson_val = pearsonr_result.statistic if not np.isnan(pearsonr_result.statistic) else 0.0
    acts_a = []
    acts_b = []
    for x in overlap_list:
        acts_a.append(info["topk_values"][idx_a][list(topk_a).index(x)])
        acts_b.append(info["topk_values"][idx_b][list(topk_b).index(x)])
    acts_a = np.array(acts_a)
    acts_b = np.array(acts_b)
    denom = np.linalg.norm(acts_a) * np.linalg.norm(acts_b)
    activation_dot = float(np.dot(acts_a, acts_b) / denom) if denom > 0 else 0.0
    return set_overlap_fraction, pearson_val, activation_dot

# def run_main_example():
#     d_model = 1536
#     n_dirs = d_model * 6
#     k = 64
#     auxk = 128
#     multik = 256
#     batch_size = 1024
#     abstract_embeddings = np.load("../data/vector_store/abstract_embeddings.npy").astype(np.float32)
#     dataloader = create_dataloader(abstract_embeddings, batch_size)
#     model_path = "checkpoints/64_9216_128_auxk_epoch_50.pth"
#     ae = load_pretrained_sae(model_path, d_model, n_dirs, k, auxk, multik)
#     info = compute_latent_info(ae, dataloader, k, multik, n_dirs)
#     plot_latent_histogram(info["latents_pre_act"], cutoff=500)
#     kvals = range(16, 256, 4)
#     example_mse = None
#     for batch_i, (batch,) in enumerate(dataloader):
#         if batch_i == 1:
#             batch = batch.to(device)
#             example_mse = compute_reconstruction_mse_over_k(ae, batch, kvals)
#             break
#     if example_mse is not None:
#         plot_mse_vs_k(kvals, example_mse, use_log_scale=True)
#     print("Completed main_example() workflow.")

def plot_log_feature_density_single(topk_indices: np.ndarray, n_dirs: int, save_path: Optional[str] = None, dpi: int = 300):
    # plot the log feature density for a single SAE's features
    num_samples, k = topk_indices.shape
    feature_counts = np.bincount(topk_indices.flatten(), minlength=n_dirs)
    feature_density = feature_counts / num_samples
    log_feature_density = np.log(feature_density + 1e-10)
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    colors = ['#FFA07A', '#20B2AA', '#778899']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(log_feature_density, kde=True, color=colors[1], edgecolor=colors[2], linewidth=1.5, alpha=0.7, ax=ax)
    ax.set_xlabel('Log Feature Density')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_feature_activation_density(feature_index: int, topk_values: np.ndarray, topk_indices: np.ndarray, save_path: Optional[str] = None, dpi: int = 300):
    # plot feature activation spectrum for a single feature
    feature_mask = topk_indices == feature_index
    activation_values = topk_values[feature_mask]
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    colors = ['#FFA07A', '#20B2AA', '#778899']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.kdeplot(activation_values, fill=True, color=colors[1], edgecolor=colors[2], linewidth=1.5, alpha=0.7, ax=ax)
    sns.histplot(activation_values, kde=False, color=colors[0], edgecolor=colors[2], linewidth=1.5, alpha=0.5, ax=ax)
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Density')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def get_embedding(text: Optional[str], client, model: str = "text-embedding-3-small", max_retries: int = 10, initial_retry_delay: int = 10, max_retry_delay: int = 120) -> Optional[np.ndarray]:
    retry_delay = initial_retry_delay
    for attempt in range(max_retries):
        embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
        return np.array(embedding, dtype=np.float32)

def search_abstracts_with_query(query: str, abstract_embeddings_path: str, abstract_texts_path: str, client, model: str = "text-embedding-3-small", k: int = 10) -> List[Tuple[str, str]]:
    abstract_embeddings = np.load(abstract_embeddings_path)
    with open(abstract_texts_path, 'r') as f:
        data = json.load(f)
    doc_ids = data['doc_ids']
    abstracts = data['abstracts']
    query_embedding = get_embedding(query, client, model=model)
    if query_embedding is None:
        return []
    sims = np.dot(abstract_embeddings, query_embedding)
    topk_indices = np.argsort(sims)[::-1][:k]
    results = []
    for idx in topk_indices:
        results.append((doc_ids[idx], abstracts[idx]))
    return results



def run_umap_on_feature_embeddings(feature_embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'cosine') -> np.ndarray:
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    return reducer.fit_transform(feature_embeddings)

def create_umap_dataframe(umap_embeddings: np.ndarray, feature_labels: List[str], feature_densities: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(umap_embeddings, columns=["UMAP1", "UMAP2"])
    df['Feature'] = feature_labels
    df['Density'] = feature_densities
    return df

# 2 different variants of umap plotting

def plot_umap_embeddings(embeddings, titles=None, start=0, end=10000, plot_title='UMAP Scatter Plot'):
    # plot umap of embeddings
    if titles is None:
        titles = [f"Point_{i}" for i in range(len(embeddings))]
    subset_emb = embeddings[start:end]
    subset_titles = titles[start:end]
    df = pd.DataFrame(subset_emb, columns=['UMAP1', 'UMAP2'])
    df['title'] = subset_titles
    fig = px.scatter(df, x='UMAP1', y='UMAP2', hover_data=['title'], title=plot_title, color_continuous_scale='Inferno')
    fig.update_traces(hovertemplate='<b>%{customdata[0]}</b>', marker={'size': 2})
    fig.update_layout(width=800, height=600)
    fig.show()

def plot_umap_scatter_log_density(df: pd.DataFrame, output_file: Optional[str] = None):
    df['log_density'] = np.log10(df['Density'] + 1e-10)
    min_log_density = df['log_density'].min()
    max_log_density = df['log_density'].max()
    fig = px.scatter(df, x='UMAP1', y='UMAP2', color='log_density', hover_data=['Feature', 'Density'], title='UMAP Scatter Plot (Log Density)', color_continuous_scale='Inferno', range_color=[min_log_density, max_log_density])
    fig.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>Density: %{customdata[1]:.4f}<br>Log Density: %{marker.color:.4f}<extra></extra>')
    fig.update_layout(width=800, height=600, coloraxis_colorbar=dict(title='Log Density', tickformat='.2f'))
    if output_file:
        fig.write_image(output_file)
    fig.show()

def create_astro_metadata_csv(dataset, output_csv: str):
    df_dict = {
        'arxiv_id': dataset['arxiv_id'],
        'title': dataset['title'],
        'citation_count': dataset['citation_count'],
        'author': dataset['author'],
        'year': dataset['year']
    }
    df = pd.DataFrame(df_dict)
    df.to_csv(output_csv, index=False)

def plot_violin_f1_token_overlap(correlations):
    # look at token overlap in autointerp explanations
    distinct_f1 = np.unique(correlations[:, 0])
    binned_data = [correlations[correlations[:, 0] == f1, 2] for f1 in distinct_f1]
    plt.violinplot(binned_data, showmedians=True)
    plt.xticks(range(1, len(distinct_f1) + 1), [round(f1, 2) for f1 in distinct_f1])
    plt.xlabel('F1 (GPT-3.5)')
    plt.ylabel('Token overlap')
    pearson_corr = scipy.stats.pearsonr(correlations[:, 0], correlations[:, 2])
    plt.show()

# def main_example_load():
#     model_64_path = 'checkpoints/16_3072_32_auxk_epoch_50.pth'
#     ae_64 = load_pretrained_sae(model_64_path, 1536*6, 1536, 64, 128, 256)
#     embeddings_path = "../data/vector_store_astroPH/abstract_embeddings.npy"
#     abstract_texts_path = "../data/vector_store_astroPH/abstract_texts.json"
#     metadata_path = "../data/vector_store_astroPH/metadata.json"
#     embeddings, abstract_texts, metadata, abstract_ids = load_embeddings_and_texts(embeddings_path, abstract_texts_path, metadata_path)
#     topk_indices_64, topk_values_64 = compute_only_topk_indices_values(ae_64, embeddings, 1024, save="sae_data_astroPH")