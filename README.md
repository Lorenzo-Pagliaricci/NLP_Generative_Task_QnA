# NLP_Generative_Task_QnA

---

## 1. Dataset Utilizzati

### Dataset 1: "rag-datasets/rag-mini-bioasq" - **question-answer-passages**
- **Formato:**  
  - **Colonne:**  
    - **question:** contiene le domande  
    - **answer:** contiene le risposte  
    - **relevant passage id:** contiene l'id del testo rilevante per la domanda

- **Esempio di formato:**

  | question | answer | relevant passage id |
  |----------|--------|---------------------|
  | . . . . | ....   | [3456, 2212, 4456]  |

### Dataset 2: "rag-datasets/rag-mini-bioasq" - **text-corpus**
- **Formato:**  
  - **Colonne:**  
    - **Passage:** contiene i testi rilevanti per le domande  
    - **id:** contiene l'id collegato al "relevant passage id" del primo dataset

- **Esempio di formato:**

  | Passage | id   |
  |---------|------|
  | ....    | 3456 |

---

## 2. Cartella: **Per_Creare_Dataset_Piccoli**

All'interno di questa cartella sono presenti tre file:

### 2.1. **Ridurre_Dimensionalità_Dataset_Reali.ipynb**
- **Funzionalità:**  
  - Scarica i dataset originali da Hugging Face.
  - Permette di selezionare il numero di righe da utilizzare.
  - Crea due dataset più piccoli con relative domande/risposte e testi associati.
- **Dataset generati:**  
  - `dataset_CONTESTI_small.parquet`
  - `dataset_Q_A_small.parquet`

### 2.2. **Creazione_DataSet_Question+Context_Answer.ipynb**
- **Funzionalità:**  
  - Prende i due dataset creati precedentemente.
  - Crea un dataset con due colonne:
    - **input_text:** contiene la domanda e i contesti associati (formattato come:  
      `question: .....`  
      `context: ......`)
    - **answer:** contiene la risposta alla domanda.
- **Formato del dataset finale:**

  | input_text                     | answer |
  |--------------------------------|--------|
  | (question e context formattati)| ....   |

- **Nome finale del dataset:**  
  - `DB_QC_A_da_utilizzare.parquet`  
  _(Questo dataset sarà usato perché i modelli da finetunare necessitano in input la domanda e i contesti associati, utili per generare la risposta.)_

### 2.3. **Creazione_DataSet_Contesti_Piccoli_Tramite_Retrieval_importato**
- **Funzionalità:**  
  - Utilizza modelli già funzionanti per eseguire un'analisi del sentiment e una ricerca incrociata sulle parole.
  - Dalla domanda vengono creati contesti più piccoli che contengono solo il necessario.
- **Dataset generato:**  
  - `DB_QC_A_retrieval.parquet`
- **Formato del dataset:**

  | retrieval_text                  | answer |
  |---------------------------------|--------|
  | (domanda e contesti ridotti)    | ....   |

---

## 3. Cartella: **BART_file**

All'interno sono presenti i codici per finetunare il modello **BART_base**:

### 3.1. **BARTbase (normale)**
- **Caratteristiche:**  
  - Non utilizza tecniche specifiche (mancano anche le metriche).
  - Ha una funzione di tokenizzazione del dataset diversa rispetto ad altri file _(da migliorare e capire come implementarla bene)_.

### 3.2. **BARTbase_with_LoRA**
- **Caratteristiche:**  
  - Utilizza tecniche di PEFT, nello specifico la tecnica **LoRA**.
  - Lo scopo è migliorare il carico computazionale e la velocità di finetuning.
  - Modifica la funzione di tokenizzazione dei dataset (train, validation e test) _(anch'essa da rivedere e migliorare)_.
  - qui sono implementate anche le metriche come rouge 1,2,3 e bleu.
- **Osservazioni:**  
  - Il modello BARTbase normale risulta più lento nell'addestramento rispetto a quello con LoRA.
  - Nonostante l'assenza delle metriche in BARTbase normale, la generazione delle risposte evidenzia una maggiore precisione in BARTbase senza LoRA.  
    _(Forse occorre addestrare meglio e di più BARTbase con LoRA e/o modificare la funzione di tokenizzazione.)_

---

## 4. Cartella: **T5_file**

- **File presente:**  
  - `T5small`
- **Caratteristiche:**  
  - Implementa il fine tuning del modello T5.
  - Al momento non sono state implementate le metriche.
  - Non è presente una versione con LoRA, anche se potrebbe non essere necessaria poiché il T5 è già veloce e leggero.
- **Osservazioni:**  
  - I parametri di training (`train_arg`) e testing (`test_arg`) sono diversi rispetto a quelli usati per BART.
  - Anche la generazione delle risposte e gli output del modello presentano differenze.
  - La generazione delle risposte non si discosta troppo da quelle vere, sebbene con BART si noti una maggiore precisione.

---

## 5. Cosa Mancano Ancora da Fare

- **Implementare le metriche** per tutti i modelli e comprendere perché alcune risultino molto basse rispetto ad altre.
- **Valutare le differenze** tra i vari modelli.
- **Verificare se il dataset retrieval** possa essere utile in fase di training con contesti più corti.
- **Implementare una vera RAG:**  
  - Addestrare un modello retriever con le domande e i contesti.
  - Valutare cosa riesce a creare autonomamente.
  - In fase di test, la domanda verrà passata prima al retriever che concatenerà i contesti giusti e successivamente tutto verrà passato al modello (quindi una RAG vera e propria).
- **Altri aspetti da valutare e implementare.**

---

