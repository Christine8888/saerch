#!/usr/bin/env python3
"""
Module: analysis_pipeline.py

Description:
    This module provides functionality for:
    1. Loading data and models for different domains (astroPH, csLG, etc.).
    2. Generating co-occurrence heatmaps for feature families.
    3. Creating large grid heatmaps for multiple families.
    4. Generating radar charts for interpretability analyses.

Author: [Your Name]
Date: [Date]
"""

import sys
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt  # If needed for your environment
# Remove or modify the following import paths based on your directory structure
sys.path.append('../saerch')
import family
import matrix
import plotly.io as pio


# =============================================================================
#                               DATA LOADING
# =============================================================================

def load_data(
    data_dir: str,
    sae_data_dir: str,
    model_paths: dict,
    autointerp_files: dict,
    batch_size: int = 1024
):
    """
    Loads embeddings, texts, and initializes DataLoader along with Family Model
    objects for multiple dimensionalities (e.g. 16, 32, 64).

    Args:
        data_dir (str): Path to directory containing `abstract_embeddings.npy` and `abstract_texts.json`.
        sae_data_dir (str): Path to the directory containing the feature analysis results.
        model_paths (dict): Dictionary of the form {'16': path_16, '32': path_32, '64': path_64}.
        autointerp_files (dict): Dictionary of the form {'16': file_16, '32': file_32, '64': file_64}
            specifying the JSON files with interpretability results.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 1024.

    Returns:
        dict: A dictionary mapping dimension to the loaded model.
        dict: A dictionary mapping dimension to all families loaded by that model.
        numpy.ndarray: The raw embeddings.
        list: The abstract texts.
    """
    # Load embeddings and texts
    abstract_embeddings = np.load(data_dir + "/abstract_embeddings.npy")
    abstract_embeddings = abstract_embeddings.astype(np.float32)
    abstract_texts = json.load(open(data_dir + '/abstract_texts.json'))['abstracts']

    # Create a DataLoader
    dataset = TensorDataset(torch.from_numpy(abstract_embeddings))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_abstracts = len(abstract_embeddings)

    # Load models and families
    models = {}
    families = {}

    for dim_key, model_path in model_paths.items():
        models[dim_key] = family.Model(
            sae_data_dir=sae_data_dir,
            model_path=model_path,
            autointerp_results=autointerp_files[dim_key],
            dataloader=dataloader,
            num_abstracts=num_abstracts
        )
        families[dim_key] = models[dim_key].load_all_families()

    return models, families, abstract_embeddings, abstract_texts


# =============================================================================
#                              HEATMAP FUNCTIONS
# =============================================================================

def make_single_heatmap(parent_family, model, display='cooc'):
    """
    Creates a single co-occurrence (or alternative) heatmap for the parent
    family. Each feature's name is displayed along both axes.

    Args:
        parent_family (family.Family): The family object for which to create the heatmap.
        model (family.Model): The model containing the similarity/co-occurrence matrix.
        display (str, optional): 'cooc' for co-occurrence or 'actsims' for activation
                                 similarity. Defaults to 'cooc'.
    """
    # Get feature IDs and names
    all_names, _ = parent_family.get_names_and_ids(model)
    # Get the normalized matrix
    norm_mat, ratio = parent_family.matrix(model, mode=display, return_mat=True)
    # Fill the diagonal with 1 for better visibility
    np.fill_diagonal(norm_mat, 1)

    heatmap = go.Heatmap(
        z=norm_mat,
        x=all_names,
        y=all_names,
        hoverongaps=False,
        hovertemplate='Name: %{x} <br> %{y} <br> %{z}'
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        width=680,
        height=600,
        yaxis=dict(scaleanchor='x'),
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.show()


def make_big_heatmap(family_names, families_dict, model, save=None, display='cooc'):
    """
    Creates a large heatmap for multiple families specified by `family_names`.

    Args:
        family_names (list): List of parent family names to include in the heatmap.
        families_dict (dict): Dictionary mapping a family name to its family object.
        model (family.Model): The model containing the similarity/co-occurrence matrix.
        save (str, optional): If provided, the path to save the figure as an image. Defaults to None.
        display (str, optional): 'cooc' for co-occurrence or 'actsims' for activation
                                 similarity. Defaults to 'cooc'.
    """
    all_ids = []

    for parent_name in family_names:
        # Extract children info
        children_info = families_dict[parent_name].children
        child_ids = [child[1] for child in children_info]
        child_names = [child[0] for child in children_info]

        # Sort children by their activation norms
        densities = [model.norms[child_id] for child_id in child_ids]
        sorted_indices = np.argsort(densities)
        sorted_ids = list(np.array(child_ids)[sorted_indices])
        # Append the parent feature ID at the end
        sorted_ids.append(model.clean_labels[parent_name]['index'])

        all_ids.extend(sorted_ids)

    # Create the matrix
    if display == 'cooc':
        min_matrix = np.minimum.outer(model.norms, model.norms)
        norm_mat = (model.mat / min_matrix)[all_ids][:, all_ids]
    else:
        # If using activation similarities
        min_matrix = 1
        norm_mat = (model.actsims / min_matrix)[all_ids][:, all_ids]

    np.fill_diagonal(norm_mat, 1)

    heatmap = go.Heatmap(
        z=norm_mat,
        hoverongaps=False,
        hovertemplate='Name: %{x} <br> %{y} <br> %{z}'
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        width=480,
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.show()

    if save:
        fig.write_image(save)


# =============================================================================
#                           RADAR CHART FUNCTIONS
# =============================================================================

def create_radar_chart(
    feature_scores,
    overall_score,
    feature_names,
    showlegend=True,
    individual_label='Individual Features',
    family_label='Family F1 (base)',
    line_color='rgb(31, 119, 180)',
    dash_color='red'
):
    """
    Creates radar chart traces for a given family (one trace for individual
    features, one trace for the overall family score).

    Args:
        feature_scores (list): List of performance metrics (e.g., F1) for individual features.
        overall_score (float): The baseline or family-level metric (e.g., family F1).
        feature_names (list): Names of the features displayed in a circular axis.
        showlegend (bool, optional): Whether to show the legend. Defaults to True.
        individual_label (str, optional): Label for the individual features trace. Defaults to 'Individual Features'.
        family_label (str, optional): Label for the overall family trace. Defaults to 'Family F1 (base)'.
        line_color (str, optional): Color for the individual features trace. Defaults to 'rgb(31, 119, 180)'.
        dash_color (str, optional): Color for the family baseline trace. Defaults to 'red'.

    Returns:
        list: A list of two Plotly Scatterpolar traces.
    """
    # Close the polygon
    categories = feature_names + [feature_names[0]]
    extended_scores = feature_scores + [feature_scores[0]]

    trace_individual = go.Scatterpolar(
        r=extended_scores,
        theta=categories,
        fill='toself',
        name=individual_label,
        line=dict(color=line_color, width=5),
        showlegend=showlegend
    )

    trace_family = go.Scatterpolar(
        r=[overall_score] * len(categories),
        theta=categories,
        name=family_label,
        line=dict(color=dash_color, width=5, dash='dash'),
        showlegend=showlegend
    )

    return [trace_individual, trace_family]


def plot_radar_charts(all_results, mode='f1'):
    """
    Plots multiple radar charts in a grid from a dictionary of family results.

    Args:
        all_results (dict): A dict where keys are family names and values
                            contain 'feature_f1', 'family_f1', and 'feature_names'.
        mode (str, optional): Which metric to plot (e.g. 'f1'). Defaults to 'f1'.
    """
    n_cols = 2
    n_rows = 2

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{'type': 'polar'}] * n_cols] * n_rows,
        subplot_titles=list(all_results.keys()),
        vertical_spacing=0.15
    )

    # For consistent legend display
    first = True

    for i, family_name in enumerate(all_results):
        row = i // n_cols + 1
        col = i % n_cols + 1
        fam_data = all_results[family_name]

        feature_scores = fam_data.get(f'feature_{mode}', [])
        overall_score = fam_data.get(f'family_{mode}', 0)
        feature_names = fam_data.get('feature_names', [])

        radar_traces = create_radar_chart(
            feature_scores,
            overall_score,
            feature_names,
            showlegend=first
        )
        first = False  # Show legend only on the first subplot

        for trace in radar_traces:
            fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        height=2000,
        width=2700,
        font=dict(family="Palatino", size=12),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="right",
            x=1,
            font=dict(size=36)
        )
    )

    # Ensure consistent radial axis
    fig.update_polars(
        radialaxis=dict(
            range=[0, 1],
            tickfont=dict(size=12),
            tickangle=45
        ),
        angularaxis=dict(
            tickfont=dict(size=17)
        )
    )

    # Adjust subplot titles
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=36, family="Palatino")
        ann['y'] = ann['y'] + 0.05

    fig.show()
    fig.write_image('radar_charts.pdf')


def plot_single_radar_chart(family_data, mode='f1', base_font_size=12):
    """
    Plots a single radar chart for one family. 
    This version includes an example of how you might rename feature names 
    or override them with a custom list.

    Args:
        family_data (dict): A dictionary with one key (the family name).
            The value must be a dict containing 'feature_names',
            'feature_f1', and 'family_f1' (if mode='f1').
        mode (str, optional): Metric to plot. Defaults to 'f1'.
        base_font_size (int, optional): Base font size for the plot. Defaults to 12.
    """
    # Extract single family name and data
    family_name = list(family_data.keys())[0].replace('*', '')
    fam_results = family_data[family_name]

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'polar'}]],
        subplot_titles=[family_name],
    )

    # Possibly override or rename feature names as needed
    original_feature_names = fam_results.get('feature_names', [])
    # Example of a custom list (shown in your code snippet):
    custom_feature_names = [
        'Numerical relativity in binary systems',
        'Third-generation gravitational wave detectors',
        'NANOGrav signal interpretations',
        'Stochastic processes',
        'Gravitational wave polarization modes',
        'Deep learning in gravitational wave detection',
        'Gravitational Waves',
        'Gravitational waves as standard sirens',
        'Gravitational wave detection technology',
        'Gravitational-wave noise reduction',
        'TianQin project',
        'KAGRA gravitational-wave observatory',
        'Gravitational waves in EMRIs and pulsars',
        'GW150914 event and LIGO observations',
        'Continuous phenomena',
        'Gravitational-wave memory effect',
        'Gravitational wave follow-up observations',
        'Gravitational waves detection methods'
    ]

    feature_scores = fam_results.get(f'feature_{mode}', [])
    overall_score = fam_results.get(f'family_{mode}', 0)

    # Generate traces
    radar_traces = create_radar_chart(
        feature_scores,
        overall_score,
        custom_feature_names,  # or original_feature_names if you don't need custom text
        showlegend=True
    )

    for trace in radar_traces:
        fig.add_trace(trace)

    fig.update_layout(
        height=1200,
        width=2550,
        font=dict(family="Palatino", size=base_font_size),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True
    )

    # Update polar axes
    fig.update_polars(
        radialaxis=dict(
            range=[0, 1],
            tickfont=dict(size=base_font_size),
            tickangle=45
        ),
        angularaxis=dict(
            tickfont=dict(size=int(base_font_size * 2.5))
        )
    )

    # Update subplot title
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=base_font_size * 3, family="Palatino")
        ann['y'] = ann['y'] + 0.05

    fig.show()
    fig.write_image('single_radar_chart.pdf')


# =============================================================================
#                              USAGE EXAMPLE
# =============================================================================

def main():
    """
    Example main function demonstrating how one might use the above code 
    to load data, instantiate models, and create plots.
    Modify the file paths and model details as needed for your environment.
    """
    # -------------------------------------------------------------------------
    # Example for astroPH domain
    # -------------------------------------------------------------------------
    astro_mode = 'epoch_astroPH_old'
    astro_SAE_DATA_DIR = '../saerch/sae_data_astroPH/'
    astro_DATA_DIR = '../data/vector_store_astroPH/'
    astro_model_paths = {
        '16': f'../models/16_3072_24_auxk_{astro_mode}_50.pth',
        '32': f'../models/32_6144_48_auxk_{astro_mode}_50.pth',
        '64': f'../models/64_9216_128_auxk_{astro_mode}_50.pth'
    }
    astro_autointerp_files = {
        '16': 'feature_analysis_results_16.json',
        '32': 'feature_analysis_results_32.json',
        '64': 'feature_analysis_results_64.json'
    }

    astro_models, astro_families, astro_embeddings, astro_texts = load_data(
        data_dir=astro_DATA_DIR,
        sae_data_dir=astro_SAE_DATA_DIR,
        model_paths=astro_model_paths,
        autointerp_files=astro_autointerp_files
    )

    # Example usage: pick a family from the '64'-dim model for a heatmap
    families_64_a = astro_families['64']
    example_family_astro = families_64_a.get('Gradient Descent Variants and Applications')
    if example_family_astro is not None:
        make_single_heatmap(example_family_astro, astro_models['64'], display='cooc')

    # Create a big heatmap for selected families
    big_heatmap_families_astro = [
        'Machine learning',
        'Gamma-Ray Bursts (GRBs)',
        'Black holes',
        'Periodicity detection methods',
        'Gas dynamics'
    ]
    make_big_heatmap(
        big_heatmap_families_astro,
        families_64_a,
        astro_models['64'],
        display='cooc',
        save='cooc_grid_astro.png'
    )

    # -------------------------------------------------------------------------
    # Example for csLG domain
    # -------------------------------------------------------------------------
    cs_mode = 'csLG_epoch'
    cs_SAE_DATA_DIR = '../saerch/sae_data_csLG/'
    cs_DATA_DIR = '../data/vector_store_csLG/'
    cs_model_paths = {
        '16': f'../models/16_3072_32_auxk_{cs_mode}_88.pth',
        '32': f'../models/32_6144_64_auxk_{cs_mode}_88.pth',
        '64': f'../models/64_9216_128_auxk_{cs_mode}_88.pth'
    }
    cs_autointerp_files = {
        '16': 'feature_analysis_results_16.json',
        '32': 'feature_analysis_results_32.json',
        '64': 'feature_analysis_results_64.json'
    }

    cs_models, cs_families, cs_embeddings, cs_texts = load_data(
        data_dir=cs_DATA_DIR,
        sae_data_dir=cs_SAE_DATA_DIR,
        model_paths=cs_model_paths,
        autointerp_files=cs_autointerp_files
    )

    # Example usage: pick a family from the '64'-dim model for a heatmap
    families_64_c = cs_families['64']
    example_family_cs = families_64_c.get('Gradient Descent Variants and Applications')
    if example_family_cs is not None:
        make_single_heatmap(example_family_cs, cs_models['64'], display='cooc')

    # Create a big heatmap for selected families
    big_heatmap_families_cs = [
        'Recurrent Neural Networks (RNNs)',
        'Explainability in machine learning and AI',
        'Recommendation Systems',
        'Deep learning in vision applications',
        'Dimensionality Reduction Techniques'
    ]
    make_big_heatmap(
        big_heatmap_families_cs,
        families_64_c,
        cs_models['64'],
        save='./cooc_grid_cs.png',
        display='cooc'
    )

    # -------------------------------------------------------------------------
    # Radar chart examples
    # -------------------------------------------------------------------------
    # Suppose we have a subset of results for multiple families
    # in astroPH dimension-16 model:
    astro_family_analysis_file = '../saerch/sae_data_astroPH/family_analysis_16_3072.json'
    with open(astro_family_analysis_file, 'r') as f:
        all_results_astro = json.load(f)

    # Convert to a dict keyed by the superfeature
    all_results_by_id_astro = {x['superfeature']: x for x in all_results_astro}

    # Example subset
    example_keys_astro = [
        'Astrophysical phenomena and interactions',
        'Cosmological Mapping Techniques',
        'Gravitational wave detection and analysis',
        'High-Energy Astrophysical Phenomena'
    ]
    example_plot_data_astro = {
        k: all_results_by_id_astro[k] for k in example_keys_astro if k in all_results_by_id_astro
    }

    # Plot multiple radar charts in a grid
    plot_radar_charts(example_plot_data_astro, mode='f1')

    # Plot single radar chart for a single family
    single_family_key = ['Gravitational wave detection and analysis']
    single_family_plot_data = {
        k: all_results_by_id_astro[k] for k in single_family_key if k in all_results_by_id_astro
    }
    plot_single_radar_chart(single_family_plot_data, base_font_size=14)


if __name__ == '__main__':
    main()