# import gradio as gr
# import numpy as np
# import json
# import pandas as pd
# from openai import OpenAI
# import yaml
# from typing import Optional, List, Dict, Tuple, Any
# from topk_sae import FastAutoencoder
# import torch
# import plotly.express as px
# from collections import Counter

# # Load configuration and initialize OpenAI client
# config = yaml.safe_load(open('../config.yaml', 'r'))
# client = OpenAI(api_key=config['jwu_openai_key'])

# EMBEDDING_MODEL = "text-embedding-3-small"

# subject = 'csLG'
# torch.set_grad_enabled(False)
# d_model = 1536
# n_dirs = d_model * 6
# k = 64
# auxk = 128
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# assert subject in ['csLG', 'astroPH']

# # Load pre-computed embeddings and texts
# embeddings_path = f"../data/vector_store_{subject}/abstract_embeddings.npy"
# texts_path = f"../data/vector_store_{subject}/abstract_texts.json"
# feature_analysis_path = f"sae_data_{subject}/feature_analysis_results_{k}.json"
# metadata_path = f'sae_data_{subject}/paper_metadata.csv'

# abstract_embeddings = np.load(embeddings_path)
# with open(texts_path, 'r') as f:
#     abstract_texts = json.load(f)
# with open(feature_analysis_path, 'r') as f:
#     feature_analysis = json.load(f)

# # Load metadata
# df_metadata = pd.read_csv(metadata_path)

# ae = FastAutoencoder(n_dirs, d_model, k, auxk, multik=0).to(device)
# if subject == 'astroPH':
#     model_path = '../models/64_9216_128_auxk_epoch_50.pth'
# elif subject == 'csLG':
#     model_path = '../models/64_9216_128_auxk_csLG_epoch_88.pth'
# ae.load_state_dict(torch.load(model_path))
# ae.eval()

# def get_embedding(text: Optional[str], model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
#     try:
#         embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
#         return np.array(embedding, dtype=np.float32)
#     except Exception as e:
#         print(f"Error getting embedding: {e}")
#         return None
    

# def intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae):
#     return ae.decode_sparse(topk_indices, topk_values)  


# # Define the custom CSS
# custom_css = """
# #custom-slider-* {
#     background-color: #ffe6e6;
# }
# """

# # Load pre-computed embeddings and texts
# folder_path = f"sae_data_{subject}/"
# data_path = f"../data/vector_store_{subject}/"

# embeddings_path = f"{data_path}abstract_embeddings.npy"
# texts_path = f"{data_path}abstract_texts.json"
# feature_analysis_path = f"{folder_path}feature_analysis_results_{k}.json"
# topk_indices_path = f"{folder_path}topk_indices_{k}_{n_dirs}.npy"
# topk_values_path = f"{folder_path}topk_values_{k}_{n_dirs}.npy"

# if subject == 'csLG':
#     weights_path = f"../models/64_9216_128_auxk_{subject}_epoch_88.pth"
# elif subject == 'astroPH':
#     weights_path = f"../models/64_9216_128_auxk_epoch_50.pth"

# abstract_embeddings = np.load(embeddings_path)
# with open(texts_path, 'r') as f:
#     abstract_texts = json.load(f)
# with open(feature_analysis_path, 'r') as f:
#     features = json.load(f)
# topk_indices = np.load(topk_indices_path)
# topk_values = np.load(topk_values_path)

# weights = torch.load(weights_path)
# decoder = weights['decoder.weight'].cpu().numpy()
# print(f"Loaded decoder with shape {decoder.shape}")
# del weights

# # Function definitions
# def get_feature_activations(abstract_texts, embeddings, topk_indices, topk_values, 
#                             m: int, min_length: int = 100, feature_index = None):
#     doc_ids = abstract_texts['doc_ids']
#     abstracts = abstract_texts['abstracts']
    
#     feature_mask = topk_indices == feature_index
#     activated_indices = np.where(feature_mask.any(axis=1))[0]
#     activation_values = np.where(feature_mask, topk_values, 0).max(axis=1)
    
#     sorted_activated_indices = activated_indices[np.argsort(-activation_values[activated_indices])]
    
#     top_m_abstracts = []
#     top_m_indices = []
#     for i in sorted_activated_indices:
#         if len(abstracts[i]) > min_length:
#             top_m_abstracts.append((doc_ids[i], abstracts[i], activation_values[i]))
#             top_m_indices.append(i)
#         if len(top_m_abstracts) == m:
#             break
    
#     return top_m_abstracts

# def calculate_co_occurrences(topk_indices, target_index, n_features=9216):
#     mask = np.any(topk_indices == target_index, axis=1)
#     co_occurring_indices = topk_indices[mask].flatten()
#     co_occurrences = Counter(co_occurring_indices)
#     del co_occurrences[target_index]
#     result = np.zeros(n_features, dtype=int)
#     result[list(co_occurrences.keys())] = list(co_occurrences.values())
#     return result


# def style_dataframe(df: pd.DataFrame, is_top: bool) -> pd.DataFrame:
#     cosine_values = df['Cosine similarity'].astype(float)
#     min_val = cosine_values.min()
#     max_val = cosine_values.max()
    
#     def color_similarity(val):
#         val = float(val)
#         # Normalize the value between 0 and 1
#         if is_top:
#             normalized_val = (val - min_val) / (max_val - min_val)
#         else:
#             # For bottom correlated, reverse the normalization
#             normalized_val = (max_val - val) / (max_val - min_val)
        
#         # Adjust the color intensity to avoid zero intensity
#         color_intensity = 0.2 + (normalized_val * 0.8)  # This ensures the range is from 0.2 to 1.0
        
#         if is_top:
#             color = f'background-color: rgba(0, 255, 0, {color_intensity:.2f})'
#         else:
#             color = f'background-color: rgba(255, 0, 0, {color_intensity:.2f})'
#         return color

#     return df.style.applymap(color_similarity, subset=['Cosine similarity'])

# def get_feature_from_index(index):
#     feature = next((f for f in features if f['index'] == index), None)
#     return feature

# def visualize_feature(index):
#     feature = next((f for f in features if f['index'] == index), None)
#     if feature is None:
#         return "Invalid feature index", None, None, None, None, None, None

#     output = f"# {feature['label']}\n\n"
#     output += f"* Pearson correlation: {feature['pearson_correlation']:.4f}\n\n"
#     output += f"* Density: {feature['density']:.4f}\n\n"

#     # Top m abstracts
#     m = 5
#     top_m_abstracts = get_feature_activations(abstract_texts, abstract_embeddings, topk_indices, topk_values, m, feature_index=index)
    
#     # Create dataframe for top abstracts
#     df_data = [
#         {"Title": m[1].split('\n\n')[0], "Activation value": f"{m[2]:.4f}"}
#         for m in top_m_abstracts
#     ]
#     df_top_abstracts = pd.DataFrame(df_data)

#     # Activation value distribution
#     activation_values = np.where(topk_indices == index, topk_values, 0).max(axis=1)
#     fig2 = px.histogram(x=activation_values, nbins=50)
#     fig2.update_layout(
#         #title=f'{feature["label"]}',
#         xaxis_title='Activation value',
#         yaxis_title=None,
#         yaxis_type='log',
#         height=220,
#     )

#     # Correlated features
#     feature_vector = decoder[:, index]
#     decoder_without_feature = np.delete(decoder, index, axis=1)
#     # Print shapes
#     print(feature_vector.shape)
#     print(decoder_without_feature.shape)
#     cosine_similarities = np.dot(feature_vector, decoder_without_feature) / (np.linalg.norm(decoder_without_feature, axis=0) * np.linalg.norm(feature_vector))

#     topk = 5
#     topk_indices_cosine = np.argsort(-cosine_similarities)[:topk]
#     topk_values_cosine = cosine_similarities[topk_indices_cosine]

#     # Create dataframe for top 5 correlated features
#     df_top_correlated = pd.DataFrame({
#         "Feature": [get_feature_from_index(i)['label'] for i in topk_indices_cosine],
#         "Cosine similarity": [f"{v:.4f}" for v in topk_values_cosine]
#     })
#     df_top_correlated_styled = style_dataframe(df_top_correlated, is_top=True)

#     bottomk = 5
#     bottomk_indices_cosine = np.argsort(cosine_similarities)[:bottomk]
#     bottomk_values_cosine = cosine_similarities[bottomk_indices_cosine]

#     # Create dataframe for bottom 5 correlated features
#     df_bottom_correlated = pd.DataFrame({
#         "Feature": [get_feature_from_index(i)['label'] for i in bottomk_indices_cosine],
#         "Cosine similarity": [f"{v:.4f}" for v in bottomk_values_cosine]
#     })
#     df_bottom_correlated_styled = style_dataframe(df_bottom_correlated, is_top=False)

#    # Co-occurrences
#     co_occurrences = calculate_co_occurrences(topk_indices, index)
#     topk = 5
#     topk_indices_co_occurrence = np.argsort(-co_occurrences)[:topk]
#     topk_values_co_occurrence = co_occurrences[topk_indices_co_occurrence]

#     # Create dataframe for top 5 co-occurring features
#     df_co_occurrences = pd.DataFrame({
#         "Feature": [get_feature_from_index(i)['label'] for i in topk_indices_co_occurrence],
#         "Co-occurrences": topk_values_co_occurrence
#     })

#     # Remove the log scale plot (fig1) from the return statement
#     return output, df_top_abstracts, df_top_correlated_styled, df_bottom_correlated_styled, df_co_occurrences, fig2


# # Create the Blocks interface with the custom CSS
# with gr.Blocks(css=custom_css) as demo:
#     with gr.Tabs():
#         with gr.Tab("SAErch"):
#             input_text = gr.Textbox(label="input")
#             search_results_state = gr.State([])
#             feature_values_state = gr.State([])
#             feature_indices_state = gr.State([])
#             manually_added_features_state = gr.State([])

#             def intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae):
#                 with torch.no_grad():
#                     return ae.decode_sparse(topk_indices, topk_values)

#             def update_search_results(feature_values, feature_indices, manually_added_features):
#                 print(f"\nEntering update_search_results")
#                 print(f"Received feature_values: {feature_values[:10]}...")
#                 print(f"Received feature_indices: {feature_indices[:10]}...")
#                 print(f"Received manually_added_features: {manually_added_features}")

#                 # Combine manually added features with query-generated features
#                 all_indices = []
#                 all_values = []
                
#                 # Add manually added features first
#                 for index in manually_added_features:
#                     if index not in all_indices:
#                         all_indices.append(index)
#                         all_values.append(feature_values[feature_indices.index(index)] if index in feature_indices else 0.0)
                
#                 # Add remaining query-generated features
#                 for index, value in zip(feature_indices, feature_values):
#                     if index not in all_indices:
#                         all_indices.append(index)
#                         all_values.append(value)

#                 print(f"Combined all_indices: {all_indices[:10]}...")
#                 print(f"Combined all_values: {all_values[:10]}...")

#                 # Reconstruct query embedding
#                 topk_indices = torch.tensor(all_indices).to(device)
#                 topk_values = torch.tensor(all_values).to(device)
                
#                 intervened_embedding = intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae)
#                 intervened_embedding = intervened_embedding.cpu().numpy().flatten()

#                 # Perform similarity search
#                 sims = np.dot(abstract_embeddings, intervened_embedding)
#                 topk_indices_search = np.argsort(sims)[::-1][:10]
#                 doc_ids = abstract_texts['doc_ids']
#                 topk_doc_ids = [doc_ids[i] for i in topk_indices_search]

#                 # Prepare search results
#                 search_results = []
#                 for doc_id in topk_doc_ids:
#                     print(f"Processing doc_id: {doc_id}")
#                     print(df_metadata.head())
#                     metadata = df_metadata[df_metadata['arxiv_id'] == doc_id].iloc[0]
#                     title = metadata['title'].replace('[', '').replace(']', '')
#                     search_results.append([
#                         title,
#                         int(metadata['citation_count']),
#                         int(metadata['year'])
#                     ])

#                 print(f"Exiting update_search_results")
#                 return search_results, all_values, all_indices

#             @gr.render(inputs=[input_text, search_results_state, feature_values_state, feature_indices_state, manually_added_features_state])
#             def show_components(text, search_results, feature_values, feature_indices, manually_added_features):
#                 print(f"Entering show_components")
#                 print(f"Input text: {text}")
#                 print(f"Received search_results: {len(search_results) if search_results else 0}")
#                 print(f"Received feature_values: {feature_values[:5] if feature_values else 'None'}...")
#                 print(f"Received feature_indices: {feature_indices[:5] if feature_indices else 'None'}...")
#                 print(f"Received manually_added_features: {manually_added_features}")

#                 if len(text) == 0:
#                     return gr.Markdown("## No Input Provided")

#                 if not search_results or text != getattr(show_components, 'last_query', None):
#                     print("New query detected, updating results and sliders")
#                     show_components.last_query = text
#                     query_embedding = get_embedding(text)

#                     with torch.no_grad():
#                         recons, z_dict = ae(torch.tensor(query_embedding).unsqueeze(0).to(device))
#                         topk_indices = z_dict['topk_indices'][0].cpu().numpy()
#                         topk_values = z_dict['topk_values'][0].cpu().numpy()

#                     feature_values = topk_values.tolist()
#                     feature_indices = topk_indices.tolist()
#                     print(f"New feature_values: {feature_values[:5]}...")
#                     print(f"New feature_indices: {feature_indices[:5]}...")
#                     search_results, feature_values, feature_indices = update_search_results(feature_values, feature_indices, manually_added_features)

#                 with gr.Row():
#                     with gr.Column(scale=2):
#                         df = gr.Dataframe(
#                             headers=["Title", "Citation Count", "Year"],
#                             value=search_results,
#                             label="Top 10 Search Results"
#                         )

#                         feature_search = gr.Textbox(label="Search Feature Labels")
#                         feature_matches = gr.CheckboxGroup(label="Matching Features", choices=[])
#                         add_button = gr.Button("Add Selected Features")

#                         def search_feature_labels(search_text):
#                             if not search_text:
#                                 return gr.CheckboxGroup(choices=[])
#                             matches = [f"{f['label']} ({f['index']})" for f in feature_analysis if search_text.lower() in f['label'].lower()]
#                             return gr.CheckboxGroup(choices=matches[:10])

#                         feature_search.change(search_feature_labels, inputs=[feature_search], outputs=[feature_matches])

#                         def on_add_features(selected_features, current_values, current_indices, manually_added_features):
#                             if selected_features:
#                                 print(f"Adding selected features: {selected_features}")
#                                 new_indices = [int(f.split('(')[-1].strip(')')) for f in selected_features]
                                
#                                 # Add new indices to manually_added_features if they're not already there
#                                 manually_added_features = list(dict.fromkeys(manually_added_features + new_indices))
                                
#                                 print(f"Updated manually_added_features: {manually_added_features}")
                                
#                                 return gr.CheckboxGroup(value=[]), current_values, current_indices, manually_added_features
#                             return gr.CheckboxGroup(value=[]), current_values, current_indices, manually_added_features


#                         add_button.click(
#                             on_add_features,
#                             inputs=[feature_matches, feature_values_state, feature_indices_state, manually_added_features_state],
#                             outputs=[feature_matches, feature_values_state, feature_indices_state, manually_added_features_state]
#                         )

#                     with gr.Column(scale=1):
#                         update_button = gr.Button("Update Results")
#                         sliders = []
#                         print(f"\nCreating sliders:")
#                         print(f"feature_values: {feature_values[:10]}...")
#                         print(f"feature_indices: {feature_indices[:10]}...")
#                         for i, (value, index) in enumerate(zip(feature_values, feature_indices)):
#                             print(f"Creating slider for feature {index} with value {value}")
#                             feature = next((f for f in feature_analysis if f['index'] == index), None)
#                             label = f"{feature['label']} ({index})" if feature else f"Feature {index}"
                            
#                             # Add prefix and change color for manually added features
#                             if index in manually_added_features:
#                                 label = f"[Custom] {label}"
#                                 slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=value, label=label, key=f"slider-{index}", elem_id=f"custom-slider-{index}")
#                             else:
#                                 slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=value, label=label, key=f"slider-{index}")
                            
#                             sliders.append(slider)


#                 def on_slider_change(*values):
#                     print("\nEntering on_slider_change")
#                     print(f"Received values: {values[:10]}...")
                    
#                     # The last value is manually_added_features
#                     manually_added_features = values[-1]
#                     slider_values = list(values[:-1])
                    
#                     # Reconstruct feature_indices based on the order of sliders
#                     reconstructed_indices = [int(slider.label.split('(')[-1].split(')')[0]) for slider in sliders]
                    
#                     new_results, new_values, new_indices = update_search_results(slider_values, reconstructed_indices, manually_added_features)
#                     print(f"New feature_values after update: {new_values[:10]}...")
#                     print(f"New feature_indices after update: {new_indices[:10]}...")
#                     print("Exiting on_slider_change")
#                     return new_results, new_values, new_indices, manually_added_features

#                 update_button.click(
#                     on_slider_change,
#                     inputs=sliders + [manually_added_features_state],
#                     outputs=[search_results_state, feature_values_state, feature_indices_state, manually_added_features_state]
#                 )

#                 print(f"Exiting show_components")
#                 return [df, feature_search, feature_matches, add_button, update_button] + sliders
            
#         with gr.Tab("Feature Visualisation"):
#             gr.Markdown("# Feature Visualiser")
#             with gr.Row():
#                 feature_search = gr.Textbox(label="Search Feature Labels")
#                 feature_matches = gr.CheckboxGroup(label="Matching Features", choices=[])
#                 visualize_button = gr.Button("Visualize Feature")
            
#             feature_info = gr.Markdown()
#             abstracts_heading = gr.Markdown("## Top 5 Abstracts")
#             top_abstracts = gr.Dataframe(
#                 headers=["Title", "Activation value"],
#                 interactive=False
#             )
            
#             gr.Markdown("## Correlated Features")
#             with gr.Row():
#                 with gr.Column(scale=1):
#                     gr.Markdown("### Top 5 Correlated Features")
#                     top_correlated = gr.Dataframe(
#                         headers=["Feature", "Cosine similarity"],
#                         interactive=False
#                     )
#                 with gr.Column(scale=1):
#                     gr.Markdown("### Bottom 5 Correlated Features")
#                     bottom_correlated = gr.Dataframe(
#                         headers=["Feature", "Cosine similarity"],
#                         interactive=False
#                     )
            
#             with gr.Row():
#                 with gr.Column(scale=1):
#                     gr.Markdown("## Top 5 Co-occurring Features")
#                     co_occurring_features = gr.Dataframe(
#                         headers=["Feature", "Co-occurrences"],
#                         interactive=False
#                     )
#                 with gr.Column(scale=1):
#                     gr.Markdown(f"## Activation Value Distribution")
#                     activation_dist = gr.Plot()

#             def search_feature_labels(search_text):
#                 if not search_text:
#                     return gr.CheckboxGroup(choices=[])
#                 matches = [f"{f['label']} ({f['index']})" for f in features if search_text.lower() in f['label'].lower()]
#                 return gr.CheckboxGroup(choices=matches[:10])

#             feature_search.change(search_feature_labels, inputs=[feature_search], outputs=[feature_matches])

#             def on_visualize(selected_features):
#                 if not selected_features:
#                     return "Please select a feature to visualize.", None, None, None, None, None, "", []
                
#                 # Extract the feature index from the selected feature string
#                 feature_index = int(selected_features[0].split('(')[-1].strip(')'))
#                 feature_info, top_abstracts, top_correlated, bottom_correlated, co_occurring_features, activation_dist = visualize_feature(feature_index)
                
#                 # Return the visualization results along with empty values for search box and checkbox
#                 return feature_info, top_abstracts, top_correlated, bottom_correlated, co_occurring_features, activation_dist, "", []

#             visualize_button.click(
#                 on_visualize,
#                 inputs=[feature_matches],
#                 outputs=[feature_info, top_abstracts, top_correlated, bottom_correlated, co_occurring_features, activation_dist, feature_search, feature_matches]
#             )

# demo.launch()

import gradio as gr
import numpy as np
import json
import pandas as pd
from openai import OpenAI
import yaml
from typing import Optional, List, Dict, Tuple, Any
from topk_sae import FastAutoencoder
import torch
import plotly.express as px
from collections import Counter

# Load configuration and initialize OpenAI client
config = yaml.safe_load(open('../config.yaml', 'r'))
client = OpenAI(api_key=config['jwu_openai_key'])

EMBEDDING_MODEL = "text-embedding-3-small"

d_model = 1536
n_dirs = d_model * 6
k = 64
auxk = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)

# Function to load data for a specific subject
def load_subject_data(subject):
    embeddings_path = f"../data/vector_store_{subject}/abstract_embeddings.npy"
    texts_path = f"../data/vector_store_{subject}/abstract_texts.json"
    feature_analysis_path = f"sae_data_{subject}/feature_analysis_results_{k}.json"
    metadata_path = f'sae_data_{subject}/paper_metadata.csv'
    topk_indices_path = f"sae_data_{subject}/topk_indices_{k}_{n_dirs}.npy"
    topk_values_path = f"sae_data_{subject}/topk_values_{k}_{n_dirs}.npy"

    abstract_embeddings = np.load(embeddings_path)
    with open(texts_path, 'r') as f:
        abstract_texts = json.load(f)
    with open(feature_analysis_path, 'r') as f:
        feature_analysis = json.load(f)
    df_metadata = pd.read_csv(metadata_path)
    topk_indices = np.load(topk_indices_path)
    topk_values = np.load(topk_values_path)

    if subject == 'astroPH':
        model_path = '../models/64_9216_128_auxk_epoch_50.pth'
    elif subject == 'csLG':
        model_path = '../models/64_9216_128_auxk_csLG_epoch_88.pth'

    ae = FastAutoencoder(n_dirs, d_model, k, auxk, multik=0).to(device)
    ae.load_state_dict(torch.load(model_path))
    ae.eval()

    weights = torch.load(model_path)
    decoder = weights['decoder.weight'].cpu().numpy()
    del weights

    return {
        'abstract_embeddings': abstract_embeddings,
        'abstract_texts': abstract_texts,
        'feature_analysis': feature_analysis,
        'df_metadata': df_metadata,
        'topk_indices': topk_indices,
        'topk_values': topk_values,
        'ae': ae,
        'decoder': decoder
    }

# Load data for both subjects
subject_data = {
    'astroPH': load_subject_data('astroPH'),
    'csLG': load_subject_data('csLG')
}

# Update existing functions to use the selected subject's data
def get_embedding(text: Optional[str], model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    try:
        embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae):
    with torch.no_grad():
        return ae.decode_sparse(topk_indices, topk_values)

# Function definitions for feature activation, co-occurrence, styling, etc.
def get_feature_activations(subject, feature_index, m=5, min_length=100):
    abstract_texts = subject_data[subject]['abstract_texts']
    abstract_embeddings = subject_data[subject]['abstract_embeddings']
    topk_indices = subject_data[subject]['topk_indices']
    topk_values = subject_data[subject]['topk_values']

    doc_ids = abstract_texts['doc_ids']
    abstracts = abstract_texts['abstracts']
    
    feature_mask = topk_indices == feature_index
    activated_indices = np.where(feature_mask.any(axis=1))[0]
    activation_values = np.where(feature_mask, topk_values, 0).max(axis=1)
    
    sorted_activated_indices = activated_indices[np.argsort(-activation_values[activated_indices])]
    
    top_m_abstracts = []
    top_m_indices = []
    for i in sorted_activated_indices:
        if len(abstracts[i]) > min_length:
            top_m_abstracts.append((doc_ids[i], abstracts[i], activation_values[i]))
            top_m_indices.append(i)
        if len(top_m_abstracts) == m:
            break
    
    return top_m_abstracts

def calculate_co_occurrences(subject, target_index, n_features=9216):
    topk_indices = subject_data[subject]['topk_indices']

    mask = np.any(topk_indices == target_index, axis=1)
    co_occurring_indices = topk_indices[mask].flatten()
    co_occurrences = Counter(co_occurring_indices)
    del co_occurrences[target_index]
    result = np.zeros(n_features, dtype=int)
    result[list(co_occurrences.keys())] = list(co_occurrences.values())
    return result

def style_dataframe(df: pd.DataFrame, is_top: bool) -> pd.DataFrame:
    cosine_values = df['Cosine similarity'].astype(float)
    min_val = cosine_values.min()
    max_val = cosine_values.max()
    
    def color_similarity(val):
        val = float(val)
        # Normalize the value between 0 and 1
        if is_top:
            normalized_val = (val - min_val) / (max_val - min_val)
        else:
            # For bottom correlated, reverse the normalization
            normalized_val = (max_val - val) / (max_val - min_val)
        
        # Adjust the color intensity to avoid zero intensity
        color_intensity = 0.2 + (normalized_val * 0.8)  # This ensures the range is from 0.2 to 1.0
        
        if is_top:
            color = f'background-color: rgba(0, 255, 0, {color_intensity:.2f})'
        else:
            color = f'background-color: rgba(255, 0, 0, {color_intensity:.2f})'
        return color

    return df.style.applymap(color_similarity, subset=['Cosine similarity'])

def get_feature_from_index(subject, index):
    feature = next((f for f in subject_data[subject]['feature_analysis'] if f['index'] == index), None)
    return feature

def visualize_feature(subject, index):
    feature = next((f for f in subject_data[subject]['feature_analysis'] if f['index'] == index), None)
    if feature is None:
        return "Invalid feature index", None, None, None, None, None, None

    output = f"# {feature['label']}\n\n"
    output += f"* Pearson correlation: {feature['pearson_correlation']:.4f}\n\n"
    output += f"* Density: {feature['density']:.4f}\n\n"

    # Top m abstracts
    top_m_abstracts = get_feature_activations(subject, index)
    
    # Create dataframe for top abstracts
    df_data = [
        {"Title": m[1].split('\n\n')[0], "Activation value": f"{m[2]:.4f}"}
        for m in top_m_abstracts
    ]
    df_top_abstracts = pd.DataFrame(df_data)

    # Activation value distribution
    topk_indices = subject_data[subject]['topk_indices']
    topk_values = subject_data[subject]['topk_values']

    activation_values = np.where(topk_indices == index, topk_values, 0).max(axis=1)
    fig2 = px.histogram(x=activation_values, nbins=50)
    fig2.update_layout(
        #title=f'{feature["label"]}',
        xaxis_title='Activation value',
        yaxis_title=None,
        yaxis_type='log',
        height=220,
    )

    # Correlated features
    decoder = subject_data[subject]['decoder']
    feature_vector = decoder[:, index]
    decoder_without_feature = np.delete(decoder, index, axis=1)
    cosine_similarities = np.dot(feature_vector, decoder_without_feature) / (np.linalg.norm(decoder_without_feature, axis=0) * np.linalg.norm(feature_vector))

    topk = 5
    topk_indices_cosine = np.argsort(-cosine_similarities)[:topk]
    topk_values_cosine = cosine_similarities[topk_indices_cosine]

    # Create dataframe for top 5 correlated features
    df_top_correlated = pd.DataFrame({
        "Feature": [get_feature_from_index(subject, i)['label'] for i in topk_indices_cosine],
        "Cosine similarity": [f"{v:.4f}" for v in topk_values_cosine]
    })
    df_top_correlated_styled = style_dataframe(df_top_correlated, is_top=True)

    bottomk = 5
    bottomk_indices_cosine = np.argsort(cosine_similarities)[:bottomk]
    bottomk_values_cosine = cosine_similarities[bottomk_indices_cosine]

    # Create dataframe for bottom 5 correlated features
    df_bottom_correlated = pd.DataFrame({
        "Feature": [get_feature_from_index(subject, i)['label'] for i in bottomk_indices_cosine],
        "Cosine similarity": [f"{v:.4f}" for v in bottomk_values_cosine]
    })
    df_bottom_correlated_styled = style_dataframe(df_bottom_correlated, is_top=False)

    # Co-occurrences
    co_occurrences = calculate_co_occurrences(subject, index)
    topk = 5
    topk_indices_co_occurrence = np.argsort(-co_occurrences)[:topk]
    topk_values_co_occurrence = co_occurrences[topk_indices_co_occurrence]

    # Create dataframe for top 5 co-occurring features
    df_co_occurrences = pd.DataFrame({
        "Feature": [get_feature_from_index(subject, i)['label'] for i in topk_indices_co_occurrence],
        "Co-occurrences": topk_values_co_occurrence
    })

    return output, df_top_abstracts, df_top_correlated_styled, df_bottom_correlated_styled, df_co_occurrences, fig2

# Modify the main interface function
def create_interface():
    custom_css = """
    #custom-slider-* {
        background-color: #ffe6e6;
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        subject = gr.Dropdown(choices=['astroPH', 'csLG'], label="Select Subject", value='astroPH')
        
        with gr.Tabs():
            with gr.Tab("SAErch"):
                input_text = gr.Textbox(label="input")
                search_results_state = gr.State([])
                feature_values_state = gr.State([])
                feature_indices_state = gr.State([])
                manually_added_features_state = gr.State([])

                def update_search_results(feature_values, feature_indices, manually_added_features, current_subject):
                    ae = subject_data[current_subject]['ae']
                    abstract_embeddings = subject_data[current_subject]['abstract_embeddings']
                    abstract_texts = subject_data[current_subject]['abstract_texts']
                    df_metadata = subject_data[current_subject]['df_metadata']

                    # Combine manually added features with query-generated features
                    all_indices = []
                    all_values = []
                    
                    # Add manually added features first
                    for index in manually_added_features:
                        if index not in all_indices:
                            all_indices.append(index)
                            all_values.append(feature_values[feature_indices.index(index)] if index in feature_indices else 0.0)
                    
                    # Add remaining query-generated features
                    for index, value in zip(feature_indices, feature_values):
                        if index not in all_indices:
                            all_indices.append(index)
                            all_values.append(value)

                    # Reconstruct query embedding
                    topk_indices = torch.tensor(all_indices).to(device)
                    topk_values = torch.tensor(all_values).to(device)
                    
                    intervened_embedding = intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae)
                    intervened_embedding = intervened_embedding.cpu().numpy().flatten()

                    # Perform similarity search
                    sims = np.dot(abstract_embeddings, intervened_embedding)
                    topk_indices_search = np.argsort(sims)[::-1][:10]
                    doc_ids = abstract_texts['doc_ids']
                    topk_doc_ids = [doc_ids[i] for i in topk_indices_search]

                    # Prepare search results
                    search_results = []
                    for doc_id in topk_doc_ids:
                        metadata = df_metadata[df_metadata['arxiv_id'] == doc_id].iloc[0]
                        title = metadata['title'].replace('[', '').replace(']', '')
                        search_results.append([
                            title,
                            int(metadata['citation_count']),
                            int(metadata['year'])
                        ])

                    return search_results, all_values, all_indices

                @gr.render(inputs=[input_text, search_results_state, feature_values_state, feature_indices_state, manually_added_features_state, subject])
                def show_components(text, search_results, feature_values, feature_indices, manually_added_features, current_subject):
                    if len(text) == 0:
                        return gr.Markdown("## No Input Provided")

                    if not search_results or text != getattr(show_components, 'last_query', None):
                        show_components.last_query = text
                        query_embedding = get_embedding(text)

                        ae = subject_data[current_subject]['ae']
                        with torch.no_grad():
                            recons, z_dict = ae(torch.tensor(query_embedding).unsqueeze(0).to(device))
                            topk_indices = z_dict['topk_indices'][0].cpu().numpy()
                            topk_values = z_dict['topk_values'][0].cpu().numpy()

                        feature_values = topk_values.tolist()
                        feature_indices = topk_indices.tolist()
                        search_results, feature_values, feature_indices = update_search_results(feature_values, feature_indices, manually_added_features, current_subject)

                    with gr.Row():
                        with gr.Column(scale=2):
                            df = gr.Dataframe(
                                headers=["Title", "Citation Count", "Year"],
                                value=search_results,
                                label="Top 10 Search Results"
                            )

                            feature_search = gr.Textbox(label="Search Feature Labels")
                            feature_matches = gr.CheckboxGroup(label="Matching Features", choices=[])
                            add_button = gr.Button("Add Selected Features")

                            def search_feature_labels(search_text):
                                if not search_text:
                                    return gr.CheckboxGroup(choices=[])
                                matches = [f"{f['label']} ({f['index']})" for f in subject_data[current_subject]['feature_analysis'] if search_text.lower() in f['label'].lower()]
                                return gr.CheckboxGroup(choices=matches[:10])

                            feature_search.change(search_feature_labels, inputs=[feature_search], outputs=[feature_matches])

                            def on_add_features(selected_features, current_values, current_indices, manually_added_features):
                                if selected_features:
                                    new_indices = [int(f.split('(')[-1].strip(')')) for f in selected_features]
                                    
                                    # Add new indices to manually_added_features if they're not already there
                                    manually_added_features = list(dict.fromkeys(manually_added_features + new_indices))
                                    
                                    return gr.CheckboxGroup(value=[]), current_values, current_indices, manually_added_features
                                return gr.CheckboxGroup(value=[]), current_values, current_indices, manually_added_features

                            add_button.click(
                                on_add_features,
                                inputs=[feature_matches, feature_values_state, feature_indices_state, manually_added_features_state],
                                outputs=[feature_matches, feature_values_state, feature_indices_state, manually_added_features_state]
                            )

                        with gr.Column(scale=1):
                            update_button = gr.Button("Update Results")
                            sliders = []
                            for i, (value, index) in enumerate(zip(feature_values, feature_indices)):
                                feature = next((f for f in subject_data[current_subject]['feature_analysis'] if f['index'] == index), None)
                                label = f"{feature['label']} ({index})" if feature else f"Feature {index}"
                                
                                # Add prefix and change color for manually added features
                                if index in manually_added_features:
                                    label = f"[Custom] {label}"
                                    slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=value, label=label, key=f"slider-{index}", elem_id=f"custom-slider-{index}")
                                else:
                                    slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=value, label=label, key=f"slider-{index}")
                                
                                sliders.append(slider)

                    def on_slider_change(*values):
                        manually_added_features = values[-1]
                        slider_values = list(values[:-1])
                        
                        # Reconstruct feature_indices based on the order of sliders
                        reconstructed_indices = [int(slider.label.split('(')[-1].split(')')[0]) for slider in sliders]
                        
                        new_results, new_values, new_indices = update_search_results(slider_values, reconstructed_indices, manually_added_features, current_subject)
                        return new_results, new_values, new_indices, manually_added_features

                    update_button.click(
                        on_slider_change,
                        inputs=sliders + [manually_added_features_state],
                        outputs=[search_results_state, feature_values_state, feature_indices_state, manually_added_features_state]
                    )

                    return [df, feature_search, feature_matches, add_button, update_button] + sliders

            with gr.Tab("Feature Visualisation"):
                gr.Markdown("# Feature Visualiser")
                with gr.Row():
                    feature_search = gr.Textbox(label="Search Feature Labels")
                    feature_matches = gr.CheckboxGroup(label="Matching Features", choices=[])
                    visualize_button = gr.Button("Visualize Feature")
                
                feature_info = gr.Markdown()
                abstracts_heading = gr.Markdown("## Top 5 Abstracts")
                top_abstracts = gr.Dataframe(
                    headers=["Title", "Activation value"],
                    interactive=False
                )
                
                gr.Markdown("## Correlated Features")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Top 5 Correlated Features")
                        top_correlated = gr.Dataframe(
                            headers=["Feature", "Cosine similarity"],
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### Bottom 5 Correlated Features")
                        bottom_correlated = gr.Dataframe(
                            headers=["Feature", "Cosine similarity"],
                            interactive=False
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Top 5 Co-occurring Features")
                        co_occurring_features = gr.Dataframe(
                            headers=["Feature", "Co-occurrences"],
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        gr.Markdown(f"## Activation Value Distribution")
                        activation_dist = gr.Plot()

                def search_feature_labels(search_text, current_subject):
                    if not search_text:
                        return gr.CheckboxGroup(choices=[])
                    matches = [f"{f['label']} ({f['index']})" for f in subject_data[current_subject]['feature_analysis'] if search_text.lower() in f['label'].lower()]
                    return gr.CheckboxGroup(choices=matches[:10])

                feature_search.change(search_feature_labels, inputs=[feature_search, subject], outputs=[feature_matches])

                def on_visualize(selected_features, current_subject):
                    if not selected_features:
                        return "Please select a feature to visualize.", None, None, None, None, None, "", []
                    
                    # Extract the feature index from the selected feature string
                    feature_index = int(selected_features[0].split('(')[-1].strip(')'))
                    feature_info, top_abstracts, top_correlated, bottom_correlated, co_occurring_features, activation_dist = visualize_feature(current_subject, feature_index)
                    
                    # Return the visualization results along with empty values for search box and checkbox
                    return feature_info, top_abstracts, top_correlated, bottom_correlated, co_occurring_features, activation_dist, "", []

                visualize_button.click(
                    on_visualize,
                    inputs=[feature_matches, subject],
                    outputs=[feature_info, top_abstracts, top_correlated, bottom_correlated, co_occurring_features, activation_dist, feature_search, feature_matches]
                )

        # Add logic to update components when subject changes
        def on_subject_change(new_subject):
            # Clear all states and return empty values for all components
            return [], [], [], [], "", [], "", [], None, None, None, None, None, None

        subject.change(
            on_subject_change,
            inputs=[subject],
            outputs=[search_results_state, feature_values_state, feature_indices_state, manually_added_features_state, 
                     input_text, feature_matches, feature_search, feature_matches, 
                     feature_info, top_abstracts, top_correlated, bottom_correlated, co_occurring_features, activation_dist]
        )

    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
