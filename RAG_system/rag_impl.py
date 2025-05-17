# Questo script implementa un sistema RAG (Retrieval-Augmented Generation)
# per rispondere a domande basate su un corpus di documenti pre-indicizzati.
# Il flusso generale è il seguente:
# 1. Caricamento del modello di embedding: Viene caricato un modello pre-addestrato
#    (BAAI/bge-base-en-v1.5) per convertire testo in vettori numerici (embeddings).
# 2. Caricamento dell'indice vettoriale: Viene caricato un indice FAISS pre-costruito
#    che contiene gli embeddings dei chunk di documenti del corpus (BioASQ).
# 3. Definizione della query: Viene specificata la domanda dell'utente.
# 4. Retrieval dei chunk pertinenti: Utilizzando la query e il modello di embedding,
#    vengono cercati nell'indice FAISS i chunk di testo più simili semanticamente alla query.
#    Viene utilizzata la ricerca per similarità coseno con punteggio.
# 5. Preparazione del contesto e del prompt: I chunk recuperati vengono combinati
#    con un prompt di sistema per fornire contesto al modello generativo.
# 6. Generazione della risposta: Viene utilizzato un modello linguistico locale (Mistral via Ollama)
#    per generare una risposta alla query, basandosi esclusivamente sul contesto fornito dai chunk recuperati.
# 7. Stampa della risposta: La risposta generata dal modello viene visualizzata.

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from dotenv import dotenv_values
import evaluate
import os
from datasets import load_from_disk

config = dotenv_values(".env")

FAISS_INDEX_PATH = "faiss_index_bioasq"  # Path all'indice FAISS pre-costruito
MODEL_NAME = config["MODEL_NAME"]
current_script_dir = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(
    current_script_dir,
    "" "..",
    "proveIgnazio_Transformers_Library",
    "models",
    f"merged_{MODEL_NAME}_for_ollama",
)


def load_vector_store(embeddings):
    """Carica l'indice vettoriale FAISS dal percorso specificato."""
    # --- Caricamento del database FAISS ---
    print(f"Caricamento dell'indice FAISS da '{FAISS_INDEX_PATH}'...")
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # Necessario per caricare indici salvati con versioni precedenti o diverse configurazioni
    )
    print("Indice FAISS caricato.")

    return vector_store


def find_most_similar(vector_store, embeddings, query):
    """
    Trova i chunk di testo più simili alla query nell'indice vettoriale.

    Args:
        vector_store: L'oggetto FAISS VectorStore caricato.
        embeddings: Il modello di embedding utilizzato.
        query: La stringa della domanda dell'utente.

    Returns:
        Una lista di tuple (Document, score) rappresentanti i chunk più simili e il loro punteggio di similarità.
    """
    # * --- Metodo alternativo: Retrieval come Langchain Retriever ---
    # retrieve = vector_store.as_retriever(
    #     search_kwargs={"k": 5}
    # )  # Configura per recuperare i primi 5 chunk più pertinenti
    # relevant_docs = retrieve.get_relevant_documents(query) # Esegue la ricerca
    # print(f"Numero di documenti recuperati (come Retriever): {len(relevant_docs)}")
    # for doc in relevant_docs:
    #     print(f"ID: {doc.metadata['source']}")
    #     print(f"Contenuto: {doc.page_content}")
    #     print("-" * 80)

    # * --- Metodo utilizzato: Retrieval per similarità coseno con punteggio ---
    # Converte la query in un embedding e cerca i vettori più simili nell'indice FAISS.
    # Restituisce i documenti (chunk) e il loro punteggio di similarità (distanza coseno).
    # Valori più bassi indicano maggiore similarità.
    print(f"\nRecupero dei chunk pertinenti per la query: '{query[:100]}...'")
    retrieve = vector_store.similarity_search_with_score_by_vector(
        embedding=embeddings.embed_query(query),
        k=15,  # Recupera i primi 15 chunk più pertinenti
    )

    print(f"\n--- Chunk Recuperati (Top {len(retrieve)}) ---")
    for i, (doc, score) in enumerate(retrieve):
        print(
            f"Chunk {i+1}: ID Sorgente: {doc.metadata['source']}, Punteggio Similarità (distanza): {score:.4f}"
        )
        # print(f"Contenuto: {doc.page_content[:200]}...") # Stampa solo l'inizio per brevità
        # print("-" * 80)
    if not retrieve:
        print("Nessun chunk recuperato.")
    print("-" * 80)

    return retrieve


# * Funzione per calcolare le metriche RAG
def compute_rag_metrics(predictions: list[str], references: list[str]):
    """
    Calcola le metriche ROUGE, BLEU e METEOR per le risposte generate rispetto ai riferimenti.
    """
    print("\nInizializzazione delle metriche ROUGE, BLEU, METEOR...")
    try:
        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
        meteor_metric = evaluate.load("meteor")
        print("Metriche caricate.")
    except Exception as e:
        print(f"Errore durante il caricamento delle metriche da 'evaluate': {e}")
        return {"error": "Failed to load metrics"}

    results = {}
    print("Calcolo ROUGE...")
    try:
        rouge_output = rouge_metric.compute(
            predictions=predictions, references=references
        )
        results.update({key: value for key, value in rouge_output.items()})
        print("ROUGE calcolato.")
    except Exception as e:
        print(f"ERRORE durante il calcolo di ROUGE: {e}")
        results["rouge_error"] = str(e)

    print("Calcolo BLEU...")
    try:
        # Per BLEU, i riferimenti possono essere una lista di liste di stringhe se ci sono più riferimenti per predizione.
        # Qui assumiamo una singola stringa di riferimento per predizione.
        bleu_output = bleu_metric.compute(
            predictions=predictions, references=[[ref] for ref in references]
        )
        results.update({key: value for key, value in bleu_output.items()})
        print("BLEU calcolato.")
    except Exception as e:
        print(f"ERRORE durante il calcolo di BLEU: {e}")
        results["bleu_error"] = str(e)

    print("Calcolo METEOR...")
    try:
        meteor_output = meteor_metric.compute(
            predictions=predictions, references=references
        )
        results.update({key: value for key, value in meteor_output.items()})
        print("METEOR calcolato.")
    except Exception as e:
        print(f"ERRORE durante il calcolo di METEOR: {e}")
        results["meteor_error"] = str(e)

    return results


def main():
    """Funzione principale che orchestra il processo RAG."""

    # Prompt di sistema per guidare il modello LLM
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
        based on snippets of text provided in context. Answer only using the context provided,
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """

    # Domanda dell'utente
    query = "What is the implication of histone lysine methylation in medulloblastoma?"

    # --- 1. Caricamento del modello di embedding ---
    print("Caricamento del modello di embedding...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"  # Assicurarsi che sia lo stesso modello usato per creare l'indice
    )
    print("Modello di embedding caricato.")

    # --- 2. Caricamento dell'indice vettoriale ---
    vector_store = load_vector_store(embeddings=embeddings)

    # --- Caricamento del modello generativo e tokenizer (una sola volta) ---
    text_generator = None
    # tokenizer_for_metrics = (
    #     None  # Necessario se compute_metrics_base fosse usato direttamente
    # )
    try:
        print(f"Caricamento del tokenizer da: {LOCAL_MODEL_PATH}")
        # Il tokenizer è usato dal pipeline, ma non direttamente per le metriche qui
        # dato che compute_rag_metrics prende stringhe.
        tokenizer_llm = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        tokenizer_for_metrics = (
            tokenizer_llm  # Se servisse per metriche che richiedono token ID
        )
        print("Tokenizer caricato.")

        print(f"Caricamento del modello da: {LOCAL_MODEL_PATH}")
        model_llm = AutoModelForSeq2SeqLM.from_pretrained(
            LOCAL_MODEL_PATH, device_map="mps"
        )
        print("Modello caricato.")

        text_generator = pipeline(
            "text2text-generation",
            model=model_llm,
            tokenizer=tokenizer_llm,
        )
        print("Pipeline di generazione testo caricata.")
    except Exception as e:
        print(f"\nErrore durante il caricamento del modello di generazione locale: {e}")
        print(
            "Verifica che il percorso LOCAL_MODEL_PATH sia corretto e contenga un modello e tokenizer validi."
        )
        return  # Esce se il caricamento del modello fallisce

    # --- Caricamento del Dataset di Test ---
    dataset_root_path = config.get("PREPARED_DATASET")
    if not dataset_root_path:
        # Fallback nel caso PREPARED_DATASET non sia in .env
        # Questo percorso è relativo alla posizione di rag_impl.py
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        default_dataset_path = os.path.join(
            current_script_dir,
            "..",
            "proveIgnazio_Transformers_Library",
            "data",
            "prepared_data",
        )
        dataset_root_path = default_dataset_path
        print(
            f"Variabile PREPARED_DATASET non trovata in .env, utilizzo il percorso di default: {dataset_root_path}"
        )

    print(f"Caricamento del dataset di test da: {dataset_root_path}")
    try:
        full_dataset = load_from_disk(dataset_root_path)
        if "test" not in full_dataset:
            print(
                f"Errore: Lo split 'test' non è stato trovato in {dataset_root_path}. Split disponibili: {list(full_dataset.keys())}"
            )
            return
        test_dataset = full_dataset["test"]
        # Per testare rapidamente, si può selezionare un subset:
        # test_dataset = full_dataset["test"].select(range(5))
        print(f"Dataset di test caricato con {len(test_dataset)} esempi.")

        # Verifica la presenza delle colonne necessarie
        required_columns = ["question", "answer"]
        if not all(col in test_dataset.column_names for col in required_columns):
            print(
                f"Errore: Colonne richieste ('question', 'answer') non trovate nel dataset di test."
            )
            print(f"Colonne disponibili: {test_dataset.column_names}")
            return

    except Exception as e:
        print(
            f"Errore durante il caricamento del dataset di test da '{dataset_root_path}': {e}"
        )
        print(
            "Assicurati che il percorso sia corretto e che il dataset sia stato salvato correttamente con `save_to_disk`."
        )
        return

    all_generated_answers = []
    all_reference_answers = []

    # --- Iterazione sul Test Set ---
    for i, example in enumerate(test_dataset):
        query = example.get("question")
        reference_answer = example.get(
            "answer"
        )  # Assumiamo che la colonna si chiami 'answer'

        if (
            not query or reference_answer is None
        ):  # reference_answer può essere una stringa vuota
            print(f"Saltato l'esempio {i+1} per mancanza di 'question' o 'answer'.")
            continue

        print(f"\n--- Elaborazione Esempio {i+1}/{len(test_dataset)} ---")
        print(f"Query: {query}")
        # print(f"Risposta di riferimento (snippet): {reference_answer[:100]}...")

        # --- 3. Recupero dei chunk più simili ---
        most_similar_chunks = find_most_similar(
            vector_store=vector_store, embeddings=embeddings, query=query
        )

        # --- 4. Preparazione del contesto per LLM ---
        context_for_llm = "\n".join(
            [chunk.page_content for chunk, _ in most_similar_chunks]
        )
        if not most_similar_chunks:  # Se non ci sono chunk, il contesto sarà vuoto
            print(
                "Attenzione: Nessun chunk pertinente trovato, il contesto per LLM sarà vuoto."
            )

        # --- 5. Generazione della risposta con modello custom ---
        generated_answer_text = (
            "ERRORE NELLA GENERAZIONE"  # Default in caso di fallimento
        )
        try:
            prompt_text = f"{SYSTEM_PROMPT}\n \
                            {context_for_llm}\n\n\
                            Question: {query}\n\n\
                            Answer: \
                        "

            print("\nGenerazione della risposta con modello locale...")
            output = text_generator(
                prompt_text,
                max_length=1000,
                num_return_sequences=1,
                # temperature=0.7,
                # early_stopping=True, # Potrebbe essere utile
            )
            generated_answer_text = output[0]["generated_text"].strip()
            print("Risposta generata (snippet):")
            print(f"{generated_answer_text[:200]}...")
            print("-" * 80)

        except Exception as e:
            print(
                f"\nErrore durante la generazione con il modello locale per l'esempio {i+1}: {e}"
            )
            # Manteniamo allineate le liste, aggiungendo un placeholder

        all_generated_answers.append(generated_answer_text)
        all_reference_answers.append(reference_answer)

    # --- Calcolo delle Metriche ---
    if all_generated_answers and all_reference_answers:
        if len(all_generated_answers) == len(all_reference_answers):
            print("\n--- Calcolo delle Metriche Finali ---")
            metrics = compute_rag_metrics(all_generated_answers, all_reference_answers)
            print("\nMetriche Complessive Calcolate sul Test Set:")
            for metric_name, score in metrics.items():
                if isinstance(score, float):
                    print(f"{metric_name}: {score:.4f}")
                else:
                    print(f"{metric_name}: {score}")
        else:
            print(
                "ATTENZIONE: Il numero di risposte generate e di riferimento non corrisponde. Le metriche non verranno calcolate."
            )
    else:
        print(
            "Nessuna risposta generata o di riferimento valida per calcolare le metriche."
        )


if __name__ == "__main__":
    main()
