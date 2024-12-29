import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import json
import wandb
from topk_sae import FastAutoencoder, train
import neuron_analyser
import family
import postprocess
import torch.optim as optim
import topk_sae
import family_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# configuration
SUBJECT = "astroPH"
DATA_DIR = Path("../data")
SAE_DATA_DIR = Path("../saerch/sae_data_" + SUBJECT)  # or sae_data_csLG for CS papers
CONFIG_PATH = Path("../config.yaml")

# Model hyperparameters
d_model = 1536  
n_dirs = d_model * 2  
k = 16
auxk = k * 2
multik = k * 4 # with multik turned off
auxk_coef = 1/32
clip_grad = 1.0
multik_coef = 0 # SET THIS TO TURN MULTIK ON OR OFF!
batch_size = 1024
lr = 1e-4
epochs = 50

def full_example():
    # load embeddings and create dataloader
    print("loading embeddings")
    embeddings = torch.from_numpy(np.load(DATA_DIR / "vector_store_" + SUBJECT + "/abstract_embeddings.npy")).float()
    dataset = TensorDataset(embeddings)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize and train the SAE
    print("initializing SAE")
    ae = FastAutoencoder(n_dirs=n_dirs, d_model=d_model, k=k, auxk=auxk, multik=multik).to(device)
    model_name = f"{k}_{n_dirs}_{auxk}_auxk_" + SUBJECT

    topk_sae.init_from_data_(ae, embeddings[:10000].to(device))
    optimizer = optim.Adam(ae.parameters(), lr=lr)

    train(ae, train_loader, optimizer, epochs, k, auxk_coef, multik_coef, clip_grad=clip_grad, model_name=model_name)

    # generate topk
    print("computing feature activations")
    topk_indices, topk_values = postprocess.compute_only_topk_indices_values(model=ae,
        embeddings=embeddings,
        batch_size=batch_size,
        save=str(SAE_DATA_DIR)
    )

    # run autointerp
    print("Running automatic interpretation...")
    analyzer = neuron_analyser.BatchNeuronAnalyzer(CONFIG_PATH)
    neuron_analyser.analyze_all_features(analyzer, f'feature_analysis_results_{k}_{n_dirs}_{auxk}.json')

    # search for feature families
    print("Finding feature families...")
    model = family.Model(
        sae_data_dir=str(SAE_DATA_DIR),
        model_path=f'../models/{model_name}_epoch_{epochs}.pth',
        autointerp_results=f'feature_analysis_results_{k}_{n_dirs}_{auxk}.json',
        dataloader=train_loader,
        num_abstracts=len(embeddings)
    )

    families = model.load_all_families(n=3)  # n=3 for depth of family tree
    
    # print 1st example family
    for family_name, family_obj in list(families.items())[:1]:
        _family = family.FeatureFamily(family_obj.parent, family_obj.children).exp_json()
        print(_family.exp_json())

    # autointerp families
    f_interp = neuron_analyser.analyze_all_features(analyzer, f'feature_analysis_results_{k}_{n_dirs}_{auxk}.json')

    # run statistics
    family_stats.get_family_stats(f_interp, families, model)

if __name__ == "__main__":
    full_example()
