from datasets import DatasetDict, load_dataset
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from tqdm import tqdm
import math


# --- Caricamento Dataset da HuggingFace ---
DATASET_HF_NAME = "enelpol/rag-mini-bioasq"
dataset = load_dataset(DATASET_HF_NAME, "text-corpus")


# --- Estrazione e preparazione dei documenti ---
documents = []
for index, doc in enumerate(dataset["test"]):
    doc = Document(
        page_content=doc["passage"],
        metadata={
            "id": index,
            "source": f'Dataset_ID_{doc["id"]}',
        },
    )
    documents.append(doc)


# --- Chunking dei documenti ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # 200 caratteri per chunk
    chunk_overlap=20,  # 20 caratteri di sovrapposizione tra i chunk
    length_function=len,  # Funzione per calcolare la lunghezza del testo
)

chunked_documents = text_splitter.split_documents(documents)
print(f"Numero di documenti originali: {len(documents)}")
print(f"Numero di chunk: {len(chunked_documents)}")


# --- Implementazione del modello di embedding ---
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


# --- Creazione efficiente del database FAISS con tqdm ---
print("Creazione del database FAISS...")

vector_store = None
batch_size = (
    128  # Dimensione del batch, puoi aggiustarla per ottimizzare memoria/velocità
)

if chunked_documents:
    # Calcola il numero totale di batch
    num_batches = math.ceil(len(chunked_documents) / batch_size)

    # Inizializza FAISS con il primo batch
    first_batch = chunked_documents[:batch_size]
    print(f"Inizializzazione FAISS con il primo batch di {len(first_batch)} chuck...")
    vector_store = FAISS.from_documents(first_batch, embeddings)
    print("Inizializzazione completata.")

    # Aggiungi i batch rimanenti con tqdm
    print(f"Aggiunta dei restanti chunk in {num_batches - 1} batch...")
    for i in tqdm(range(1, num_batches), desc="Adding batches to FAISS"):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(chunked_documents))
        batch = chunked_documents[start_index:end_index]
        if batch:  # Assicurati che il batch non sia vuoto
            vector_store.add_documents(batch)

    print("Database FAISS creato con successo.")
else:
    print("Nessun documento chunkizzato da aggiungere all'indice.")


# --- Salvataggio del database FAISS ---
if vector_store:  # Salva solo se l'indice è stato creato
    FAISS_INDEX_PATH = "faiss_index_bioasq"
    print(f"Salvataggio del database FAISS in {FAISS_INDEX_PATH}")
    vector_store.save_local(FAISS_INDEX_PATH)
    print("Database FAISS salvato con successo.")
else:
    print("Salvataggio saltato: l'indice FAISS non è stato creato.")
