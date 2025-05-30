# 🧠 Retrieval-Augmented Generation (RAG) per Question Answering Biomedico

Questo modulo implementa un sistema **Retrieval-Augmented Generation (RAG)** progettato per incrementare l'accuratezza e la specificità delle risposte in un contesto **biomedico**, combinando il recupero semantico di informazioni con modelli generativi fine-tunati.

## ⚙️ Architettura del Sistema

Il sistema RAG è suddiviso in due componenti principali:

### 1. Retrieval (Recupero)

- Estrae dinamicamente i **chunk di testo più rilevanti** rispetto alla domanda.
- Utilizza un indice **FAISS** costruito su embedding semantici pre-addestrati:
  - 🔗 Modello usato: [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5)
- La ricerca è basata su **similarità coseno** tra embedding.

### 2. Generazione

- Produce risposte condizionate solo sul **contesto recuperato**.
- Utilizza modelli generativi come **BART** o **Gemma**, ottimizzati tramite **LoRA (Low-Rank Adaptation)**.
- Funziona completamente in locale su CPU/GPU MPS o CUDA.

---

## 📁 Struttura dei File

### `vector_store_impl.py` – Creazione dell’indice FAISS

Trasforma il corpus testuale in un indice vettoriale interrogabile.

#### Flusso operativo:
- Caricamento del dataset `enelpol/rag-mini-bioasq` da Hugging Face.
- Conversione dei passaggi in oggetti `Document` con metadati.
- Suddivisione dei testi in chunk (200 caratteri, overlap 20).
- Generazione embedding con `BAAI/bge-base-en-v1.5`.
- Creazione dell’indice FAISS e salvataggio locale:
  
```python
  vector_store = FAISS.from_documents(chunked_documents, embeddings)
  vector_store.save_local("faiss_index_bioasq")
```


## 🔁 `rag_impl.py` – Retrieval + Generazione

Script principale che orchestra il flusso RAG:

### 🔄 Flusso operativo

1. **Caricamento embedding model** e **indice FAISS**.
2. **Embedding della query** + ricerca dei top-`k` chunk più simili.
3. **Composizione del prompt** con i chunk recuperati.
4. **Generazione della risposta** con modello generativo fine-tunato.

```python
retrieve = vector_store.similarity_search_with_score_by_vector(
    embedding=embeddings.embed_query(query), k=15
)
output = text_generator(prompt_text, max_length=1000)
```

# 🧠 Retrieval-Augmented Generation (RAG) – Sistema Biomedico

Sistema che unisce **retrieval semantico** e **generazione condizionata** per rispondere a domande in ambito biomedico con maggiore accuratezza e specificità.

---

## 🧪 Fine-tuning del Modello Generativo

Effettuato con [🤗 Transformers](https://huggingface.co/transformers) e tecnica PEFT (LoRA), per ridurre il numero di parametri aggiornati.

### 🔧 Hyperparametri principali

- `learning_rate = 1e-5`
- `train_batch_size = 5`
- `eval_batch_size = 1`
- `gradient_accumulation_steps = 2`
- `num_train_epochs = 10`
- `eval_steps = 50`
- `load_best_model_at_end = True`

### 🔩 Configurazione LoRA

- **Moduli target:** `q`, `k`, `v`, `o`
- `lora_alpha = 40`
- `r = 2`
- `lora_dropout = 0.5`

---

## 📈 Valutazione delle Prestazioni

### 📌 Valutazione della Generazione (QA)

| Metrica   | Caratteristiche                                                                 |
|-----------|----------------------------------------------------------------------------------|
| ROUGE-L   | Sequenza comune più lunga (coerenza strutturale e contenutistica)              |
| METEOR    | Riconosce sinonimi/parafrasi; valuta anche la struttura grammaticale           |
| BLEU      | Precisione sugli n-grammi (meno adatta per QA con risposte parafrasate)         |

### 📌 Valutazione del Retrieval

- **Recall@k**: verifica la presenza dei passaggi rilevanti nei top-`k` risultati.
- Misura la qualità dell’indice FAISS e degli embedding semantici.

---

## 🧩 Integrazione e Modularità

Sistema suddiviso in moduli indipendenti:

| Script               | Funzione                                                              |
|----------------------|-----------------------------------------------------------------------|
| `vector_store_impl.py` | Costruisce l’indice FAISS a partire dal corpus testuale              |
| `rag_impl.py`        | Interroga l’indice e genera la risposta                               |
| `metrics_utils.py`   | Calcola ROUGE, BLEU, METEOR per la valutazione automatica             |

---

## ✅ Requisiti e Setup

Assicurati di avere installato i seguenti pacchetti:

```bash
pip install transformers datasets peft faiss-cpu accelerate evaluate langchain
```

## 🛠 Esecuzione

### Costruzione dell’indice (solo al primo utilizzo)

```bash
python vector_store_impl.py
```

### 🔍 Generazione della risposta per una query

Esegui il comando seguente per generare una risposta condizionata al contesto recuperato:

```bash
python rag_impl.py --query "What is the treatment for melanoma?"
```

❗ Considerazioni Finali  
Il sistema migliora la precisione e la specificità delle risposte nel dominio biomedico.

È stato ottimizzato per Apple Silicon (MPS), ma è compatibile anche con GPU CUDA.

Tutti gli script sono modulari, documentati e facilmente riutilizzabili per esperimenti e adattamenti futuri.
