from datasets import DatasetDict, load_dataset
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

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
# Assicurati di avere il server Ollama in esecuzione
embeddings = OllamaEmbeddings(
    model="BAAI/bge-base-en-v1.5",
    base_url="http://localhost:11434",
)


# --- Creazione del database FAISS ---
# Crea un database FAISS per l'archiviazione dei chunk
vector_store = FAISS.from_documents(
    chunked_documents,
    embeddings,
)
