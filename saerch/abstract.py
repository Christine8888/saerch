import numpy as np
import pickle
import json
from typing import List
from dataclasses import dataclass

import argparse

parser = argparse.ArgumentParser(description="Abstract processor")
parser.add_argument("--subject", type=str, default="astroPH", help="ID of the vector store")
args = parser.parse_args()
SUBJECT = args.subject

@dataclass
class Document:
    id: str
    abstract: str
    conclusions: str
    arxiv_id: str
    title: str = None
    score: float = None
    n_citation: int = None
    keywords: List[str] = None

# load the full embeddings matrix
embeddings = np.load(f'../data/vector_store_{SUBJECT}/embeddings_matrix.npy')

# load the index mapping
with open(f'../data/vector_store_{SUBJECT}/index_mapping.pkl', 'rb') as f:
    index_mapping = pickle.load(f)

# load the documents
with open(f'../data/vector_store_{SUBJECT}/documents.pkl', 'rb') as f:
    documents = pickle.load(f)

# create lists to store the abstract embeddings and corresponding text
abstract_embeddings = []
abstract_texts = []
doc_ids = []

# iterate through the index mapping to find abstract indices and corresponding text
for doc_id, mappings in index_mapping.items():
    if 'abstract' in mappings:
        abstract_index = mappings['abstract']
        abstract_embeddings.append(embeddings[abstract_index])
        
        # find the corresponding document and extract the abstract text
        doc = next((d for d in documents if d.id == doc_id), None)
        if doc:
            abstract_texts.append(doc.abstract)
            doc_ids.append(doc_id)
        else:
            print(f"warning: document with id {doc_id} not found.")

# convert lists to numpy arrays
abstract_embeddings = np.array(abstract_embeddings)

print(f"Processed {len(abstract_embeddings)} abstract embeddings.")
print(f"Shape of abstract embeddings: {abstract_embeddings.shape}")
print(f"Length of abstract texts: {len(abstract_texts)}")
print(f"Length of document ids: {len(doc_ids)}")

# save the abstract embeddings
np.save(f'../data/vector_store_{SUBJECT}/abstract_embeddings.npy', abstract_embeddings)

# save the abstract texts and document ids
with open(f'../data/vector_store_{SUBJECT}/abstract_texts.json', 'w') as f:
    json.dump({
        'doc_ids': doc_ids,
        'abstracts': abstract_texts
    }, f)

print("Saved abstract embeddings and texts.")
