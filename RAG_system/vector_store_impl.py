# Questo script implementa il processo di creazione di un Vector Store FAISS
# partendo da un dataset testuale caricato da Hugging Face.
# Il flusso principale è il seguente:
# 1. Caricamento del dataset: Viene scaricato un dataset specifico ("enelpol/rag-mini-bioasq") da Hugging Face.
# 2. Estrazione e preparazione dei documenti: I passaggi testuali del dataset vengono estratti
#    e trasformati in oggetti `Document` di LangChain, arricchiti con metadati (ID e sorgente).
# 3. Chunking dei documenti: I documenti vengono suddivisi in chunk più piccoli utilizzando
#    `RecursiveCharacterTextSplitter` per gestire meglio testi lunghi e migliorare l'efficacia
#    del recupero di informazioni.
# 4. Inizializzazione del modello di embedding: Viene caricato un modello di embedding pre-addestrato
#    ("BAAI/bge-base-en-v1.5") da Hugging Face per convertire i chunk di testo in vettori numerici.
# 5. Creazione del Vector Store FAISS: I chunk vengono processati in batch per creare efficientemente
#    un indice FAISS. FAISS è una libreria per la ricerca efficiente di similarità e clustering
#    di vettori densi. L'elaborazione in batch aiuta a gestire la memoria.
# 6. Salvataggio del Vector Store: L'indice FAISS creato viene salvato localmente su disco
#    per poter essere riutilizzato successivamente senza dover ripetere i passaggi precedenti.

from datasets import DatasetDict, load_dataset
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import math


# --- 1. Caricamento Dataset da HuggingFace ---
DATASET_HF_NAME = "enelpol/rag-mini-bioasq"  # Nome del dataset su Hugging Face Hub
print(f"Caricamento dataset: {DATASET_HF_NAME}...")
dataset = load_dataset(
    DATASET_HF_NAME, "text-corpus"
)  # Carica la configurazione 'text-corpus' del dataset
print("Dataset caricato.")


# --- 2. Estrazione e preparazione dei documenti ---
print("Estrazione e preparazione dei documenti LangChain...")
documents = []
# Itera su ogni elemento nella partizione 'test' del dataset
for index, doc_data in enumerate(dataset["test"]):
    # Crea un oggetto Document di LangChain per ogni passaggio
    doc = Document(
        page_content=doc_data["passage"],  # Il contenuto testuale del documento
        metadata={  # Metadati associati al documento
            "id": index,  # Un ID progressivo locale
            "source": f'Dataset_ID_{doc_data["id"]}',  # Un riferimento all'ID originale nel dataset
        },
    )
    documents.append(doc)
print(f"Numero totale di documenti estratti: {len(documents)}")


# --- 3. Chunking dei documenti ---
print("Suddivisione dei documenti in chunk...")
# NOTE: Esistono diverse strategie di chunking (per frasi, paragrafi, token, ecc.).
# Qui usiamo RecursiveCharacterTextSplitter che cerca di dividere su separatori specifici
# (newline, spazi, ecc.) mantenendo la coerenza semantica per quanto possibile.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Dimensione massima (in caratteri) di ogni chunk
    chunk_overlap=20,  # Numero di caratteri sovrapposti tra chunk consecutivi per mantenere il contesto
    length_function=len,  # Funzione usata per misurare la lunghezza del testo (default è len)
)

chunked_documents = text_splitter.split_documents(documents)
print(f"Numero di documenti originali: {len(documents)}")
print(f"Numero totale di chunk generati: {len(chunked_documents)}")


# --- 4. Implementazione del modello di embedding ---
# Carica il modello di embedding specificato da Hugging Face.
# "BAAI/bge-base-en-v1.5" è un modello popolare per la generazione di embedding testuali.
print("Caricamento del modello di embedding BAAI/bge-base-en-v1.5...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
print("Modello di embedding caricato.")


# --- 5. Creazione efficiente del database FAISS con tqdm ---
print("Creazione del database vettoriale FAISS...")

vector_store = None
batch_size = 128  # Numero di chunk da processare in ogni batch. Regolare per bilanciare uso di memoria e velocità.

if chunked_documents:
    # Calcola il numero totale di batch necessari
    num_batches = math.ceil(len(chunked_documents) / batch_size)
    print(
        f"Numero totale di chunk: {len(chunked_documents)}, Dimensione batch: {batch_size}, Numero batch: {num_batches}"
    )

    # Inizializza FAISS con il primo batch di documenti
    first_batch = chunked_documents[:batch_size]
    print(f"Inizializzazione FAISS con il primo batch di {len(first_batch)} chunk...")
    # `FAISS.from_documents` calcola gli embedding e costruisce l'indice iniziale
    vector_store = FAISS.from_documents(first_batch, embeddings)
    print("Indice FAISS inizializzato.")

    # Aggiunge i batch rimanenti all'indice esistente, mostrando una barra di progresso
    print(f"Aggiunta dei restanti chunk in {num_batches - 1} batch...")
    # Itera sui batch successivi al primo
    for i in tqdm(range(1, num_batches), desc="Aggiunta batch a FAISS"):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(chunked_documents))
        batch = chunked_documents[start_index:end_index]
        if (
            batch
        ):  # Assicurati che il batch non sia vuoto (importante per l'ultimo batch)
            # `add_documents` calcola gli embedding per il batch e li aggiunge all'indice esistente
            vector_store.add_documents(batch)

    print("Creazione del database FAISS completata.")
else:
    print("Nessun documento chunkizzato trovato. Impossibile creare l'indice FAISS.")


# --- 6. Salvataggio del database FAISS ---
if vector_store:  # Salva l'indice solo se è stato creato con successo
    FAISS_INDEX_PATH = (
        "faiss_index_bioasq"  # Path dove salvare l'indice FAISS localmente
    )

    print(f"Salvataggio dell'indice FAISS in '{FAISS_INDEX_PATH}'...")
    vector_store.save_local(FAISS_INDEX_PATH)
    print("Indice FAISS salvato con successo.")
else:
    print("Salvataggio saltato: l'indice FAISS non è stato creato o è vuoto.")
