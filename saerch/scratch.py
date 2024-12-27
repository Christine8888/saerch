import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
import yaml
import time
import tiktoken

# Configuration
DATASET_NAME = "charlieoneill/csLG" #"JSALT2024-Astro-LLMs/astro_paper_corpus"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS = 8192
MAX_RETRIES = 5
RETRY_DELAY = 20
BATCH_SIZE = 100
OUTPUT_DIR = "../data/vector_store_csLG"
SAVE_INTERVAL = 10000  # Save every 25000 steps

@dataclass
class Document:
    id: str
    title: str
    abstract: str
    #conclusions: str

class EmbeddingClient:
    def __init__(self, client: OpenAI, model: str = EMBEDDING_MODEL):
        self.client = client
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def truncate_text(self, text: str) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) > MAX_TOKENS:
            return self.tokenizer.decode(tokens[:MAX_TOKENS])
        return text

    def embed(self, text: str) -> np.ndarray:
        text = self.truncate_text(text.replace("\n", " "))
        for attempt in range(MAX_RETRIES):
            try:
                embedding = self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
                return np.array(embedding, dtype=np.float32)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to get embedding after {MAX_RETRIES} attempts: {e}")
                    return np.zeros(1536, dtype=np.float32)
                print(f"Error getting embedding (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {RETRY_DELAY} seconds.")
                time.sleep(RETRY_DELAY)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        truncated_texts = [self.truncate_text(text.replace("\n", " ")) for text in texts]
        for attempt in range(MAX_RETRIES):
            try:
                embeddings = self.client.embeddings.create(input=truncated_texts, model=self.model).data
                return [np.array(embedding.embedding, dtype=np.float32) for embedding in embeddings]
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to get batch embeddings after {MAX_RETRIES} attempts: {e}")
                    return [np.zeros(1536, dtype=np.float32) for _ in texts]
                print(f"Error getting batch embeddings (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {RETRY_DELAY} seconds.")
                time.sleep(RETRY_DELAY)

def load_dataset_documents() -> List[Document]:
    dataset = load_dataset(DATASET_NAME, split="train")
    documents = []
    for paper in tqdm(dataset, desc="Loading documents"):
        documents.append(Document(
            id=paper['id'],
            title=paper['title'],
            abstract=f"{paper['title']}\n\n{paper['abstract']}",
            #conclusions=f"{paper['title']}\n\n{paper['conclusions']}"
        ))
    print(f"Total documents: {len(documents)}")
    return documents

def process_embeddings(documents: List[Document], embedding_client: EmbeddingClient) -> Tuple[np.ndarray, Dict[str, Dict[str, int]], Dict[str, Document]]:
    embeddings = []
    index_mapping = {}
    document_index = {}
    current_index = 0

    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Processing embeddings"):
        batch = documents[i:i+BATCH_SIZE]
        
        abstracts = [doc.abstract for doc in batch]
        #conclusions = [doc.conclusions for doc in batch]
        
        abstract_embeddings = embedding_client.embed_batch(abstracts)
        #conclusion_embeddings = embedding_client.embed_batch(conclusions)
        
        for j, doc in enumerate(batch):
            document_index[doc.id] = doc
            index_mapping[doc.id] = {
                'abstract': current_index,
                #'conclusions': current_index + 1
            }
            embeddings.extend([abstract_embeddings[j]])#, conclusion_embeddings[j]])
            current_index += 1

        # Save data at intervals
        #print(f"Saving data at step {i + BATCH_SIZE}...")
        if (i + BATCH_SIZE) % SAVE_INTERVAL == 0 or (i + BATCH_SIZE) >= len(documents):
            print(f"Saving data at step {i + BATCH_SIZE}...")
            save_data(documents[:i+BATCH_SIZE], document_index, np.array(embeddings), index_mapping)

    return np.array(embeddings), index_mapping, document_index

def save_data(documents: List[Document], document_index: Dict[str, Document], embeddings_matrix: np.ndarray, index_mapping: Dict[str, Dict[str, int]]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "documents.pkl"), 'wb') as f:
        pickle.dump(documents, f)
    with open(os.path.join(OUTPUT_DIR, "document_index.pkl"), 'wb') as f:
        pickle.dump(document_index, f)
    np.save(os.path.join(OUTPUT_DIR, "embeddings_matrix.npy"), embeddings_matrix)
    with open(os.path.join(OUTPUT_DIR, "index_mapping.pkl"), 'wb') as f:
        pickle.dump(index_mapping, f)

def main():
    # Load configuration
    config = yaml.safe_load(open('../config.yaml', 'r'))
    client = OpenAI(api_key=config['openai_api_key'])
    embedding_client = EmbeddingClient(client)

    # Load documents
    print("Loading documents...")
    documents = load_dataset_documents()

    # Process embeddings
    print("Processing embeddings...")
    embeddings_matrix, index_mapping, document_index = process_embeddings(documents, embedding_client)

    # Final save
    print("Saving final data...")
    save_data(documents, document_index, embeddings_matrix, index_mapping)

    print("Vector store created successfully!")
    print(f"Total documents: {len(documents)}")
    print(f"Total embeddings: {len(embeddings_matrix)}")

if __name__ == "__main__":
    main()