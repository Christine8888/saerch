import sys
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader, TensorDataset
import plotly.io as pio
sys.path.append('../saerch') # or wherever saerch is
import family

def load_data(data_dir, sae_data_dir, model_paths, autointerp_files, batch_size=1024):
    # load data and all models for the analysis
    embeddings = np.load(f"{data_dir}/abstract_embeddings.npy").astype(np.float32)
    texts = json.load(open(f"{data_dir}/abstract_texts.json"))['abstracts']
    dataloader = DataLoader(TensorDataset(torch.from_numpy(embeddings)), batch_size=batch_size)
    models, families = {}, {}

    for dim, path in model_paths.items():
        models[dim] = family.Model(
            sae_data_dir=sae_data_dir,
            model_path=path,
            autointerp_results=autointerp_files[dim],
            dataloader=dataloader,
            num_abstracts=len(embeddings)
        )
        families[dim] = models[dim].load_all_families()
    
    return models, families, embeddings, texts

def make_heatmap(parent_family, model, display='cooc', single=True, save=None):
    # make family heatmap
    names, _ = parent_family.get_names_and_ids(model)
    norm_mat, _ = parent_family.matrix(model, mode=display, return_mat=True)
    np.fill_diagonal(norm_mat, 1)
    heatmap = go.Heatmap(z=norm_mat, x=names, y=names, hovertemplate='Name: %{x} <br> %{y} <br> %{z}')
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        width=680 if single else 480,
        height=600 if single else 400,
        yaxis=dict(scaleanchor='x') if single else {},
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    if not single:
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(showticklabels=False)
    fig.show()
    if save:
        fig.write_image(save)

def make_big_heatmap(family_names, families, model, display='cooc', save=None):
    # make multi family (block diagonalized) heatmap
    all_ids = []
    for name in family_names:
        children = families[name].children
        sorted_ids = sorted([child[1] for child in children], key=lambda cid: model.norms[cid]) + [model.clean_labels[name]['index']]
        all_ids.extend(sorted_ids)
    norm_mat = (model.mat / np.minimum.outer(model.norms, model.norms) if display == 'cooc' else model.actsims)[all_ids][:, all_ids]
    np.fill_diagonal(norm_mat, 1)
    heatmap = go.Heatmap(z=norm_mat, hovertemplate='Name: %{x} <br> %{y} <br> %{z}')
    fig = go.Figure(data=[heatmap])
    fig.update_layout(width=480, height=400, margin=dict(l=20, r=20, t=20, b=20))
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.show()
    if save:
        fig.write_image(save)

# make radar charts
def create_radar_traces(scores, overall, names, labels=('Individual Features', 'Family F1 (base)'), colors=('rgb(31,119,180)', 'red')):
    categories = names + [names[0]]
    return [
        go.Scatterpolar(r=scores + [scores[0]], theta=categories, fill='toself', name=labels[0],
                       line=dict(color=colors[0], width=5)),
        go.Scatterpolar(r=[overall]*len(categories), theta=categories, name=labels[1],
                       line=dict(color=colors[1], width=5, dash='dash'))
    ]

def plot_radar_charts(results, mode='f1', n_cols=2, n_rows=2, save='radar_charts.pdf'):
    fig = make_subplots(
        rows=n_rows, cols=n_cols, specs=[[{'type': 'polar'}]*n_cols]*n_rows,
        subplot_titles=list(results.keys()), vertical_spacing=0.15
    )
    first = True
    for i, (name, data) in enumerate(results.items()):
        traces = create_radar_traces(
            data.get(f'feature_{mode}', []),
            data.get(f'family_{mode}', 0),
            data.get('feature_names', []),
            showlegend=first
        )
        first = False
        row, col = i // n_cols + 1, i % n_cols + 1
        for trace in traces:
            fig.add_trace(trace, row=row, col=col)
    fig.update_layout(
        height=2000, width=2700, font=dict(family="Palatino", size=12),
        paper_bgcolor='white', plot_bgcolor='white', showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1, font=dict(size=36))
    )
    fig.update_polars(
        radialaxis=dict(range=[0, 1], tickfont=dict(size=12), tickangle=45),
        angularaxis=dict(tickfont=dict(size=17))
    )
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=36, family="Palatino")
        ann['y'] += 0.05
    fig.show()
    fig.write_image(save)

def plot_single_radar_chart(family_data, mode='f1', custom_names=None, base_font=12, save='single_radar_chart.pdf'):
    name, data = next(iter(family_data.items()))
    names = custom_names or data.get('feature_names', [])
    traces = create_radar_traces(
        data.get(f'feature_{mode}', []),
        data.get(f'family_{mode}', 0),
        names
    )
    fig = make_subplots(specs=[[{'type': 'polar'}]], subplot_titles=[name.replace('*', '')])
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(
        height=1200, width=2550, font=dict(family="Palatino", size=base_font),
        paper_bgcolor='white', plot_bgcolor='white', showlegend=True
    )
    fig.update_polars(
        radialaxis=dict(range=[0, 1], tickfont=dict(size=base_font), tickangle=45),
        angularaxis=dict(tickfont=dict(size=int(base_font * 2.5)))
    )
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=base_font * 3, family="Palatino")
        ann['y'] += 0.05
    fig.show()
    fig.write_image(save)

# Usage Example
def main():
    # AstroPH Domain
    astro_paths = {
        'sae_data': '../saerch/sae_data_astroPH/',
        'data': '../data/vector_store_astroPH/',
        'models': {
            '16': '../models/16_3072_24_auxk_epoch_astroPH_old_50.pth',
            '32': '../models/32_6144_48_auxk_epoch_astroPH_old_50.pth',
            '64': '../models/64_9216_128_auxk_epoch_astroPH_old_50.pth'
        },
        'autointerp': {
            '16': 'feature_analysis_results_16.json',
            '32': 'feature_analysis_results_32.json',
            '64': 'feature_analysis_results_64.json'
        }
    }
    astro_models, astro_families, _, _ = load_data(
        data_dir=astro_paths['data'],
        sae_data_dir=astro_paths['sae_data'],
        model_paths=astro_paths['models'],
        autointerp_files=astro_paths['autointerp']
    )
    fam_64_astro = astro_families['64'].get('Gradient Descent Variants and Applications')
    if fam_64_astro:
        make_heatmap(fam_64_astro, astro_models['64'], single=True, display='cooc')
    big_families_astro = ['Machine learning', 'Gamma-Ray Bursts (GRBs)', 'Black holes', 'Periodicity detection methods', 'Gas dynamics']
    make_big_heatmap(big_families_astro, astro_families['64'], astro_models['64'], display='cooc', save='cooc_grid_astro.png')

    # csLG Domain
    cs_paths = {
        'sae_data': '../saerch/sae_data_csLG/',
        'data': '../data/vector_store_csLG/',
        'models': {
            '16': '../models/16_3072_32_auxk_csLG_epoch_88.pth',
            '32': '../models/32_6144_64_auxk_csLG_epoch_88.pth',
            '64': '../models/64_9216_128_auxk_csLG_epoch_88.pth'
        },
        'autointerp': {
            '16': 'feature_analysis_results_16.json',
            '32': 'feature_analysis_results_32.json',
            '64': 'feature_analysis_results_64.json'
        }
    }
    cs_models, cs_families, _, _ = load_data(
        data_dir=cs_paths['data'],
        sae_data_dir=cs_paths['sae_data'],
        model_paths=cs_paths['models'],
        autointerp_files=cs_paths['autointerp']
    )
    fam_64_cs = cs_families['64'].get('Gradient Descent Variants and Applications')
    if fam_64_cs:
        make_heatmap(fam_64_cs, cs_models['64'], single=True, display='cooc')
    big_families_cs = ['Recurrent Neural Networks (RNNs)', 'Explainability in machine learning and AI', 'Recommendation Systems', 'Deep learning in vision applications', 'Dimensionality Reduction Techniques']
    make_big_heatmap(big_families_cs, cs_families['64'], cs_models['64'], display='cooc', save='./cooc_grid_cs.png')

    # example radar charts
    astro_analysis_file = '../saerch/sae_data_astroPH/family_analysis_16_3072.json'
    with open(astro_analysis_file) as f:
        all_results_astro = {x['superfeature']: x for x in json.load(f)}
    selected_astro = ['Astrophysical phenomena and interactions', 'Cosmological Mapping Techniques', 'Gravitational wave detection and analysis', 'High-Energy Astrophysical Phenomena']
    plot_radar_charts({k: all_results_astro[k] for k in selected_astro if k in all_results_astro}, mode='f1')
    custom_names = [
        'Numerical relativity in binary systems', 'Third-generation gravitational wave detectors',
        'NANOGrav signal interpretations', 'Stochastic processes', 'Gravitational wave polarization modes',
        'Deep learning in gravitational wave detection', 'Gravitational Waves', 'Gravitational waves as standard sirens',
        'Gravitational wave detection technology', 'Gravitational-wave noise reduction', 'TianQin project',
        'KAGRA gravitational-wave observatory', 'Gravitational waves in EMRIs and pulsars', 'GW150914 event and LIGO observations',
        'Continuous phenomena', 'Gravitational-wave memory effect', 'Gravitational wave follow-up observations',
        'Gravitational waves detection methods'
    ]
    plot_single_radar_chart(
        {'Gravitational wave detection and analysis': all_results_astro.get('Gravitational wave detection and analysis', {})},
        custom_names=custom_names,
        base_font=14
    )

if __name__ == '__main__':
    main()