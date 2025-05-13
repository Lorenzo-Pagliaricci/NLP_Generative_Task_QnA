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


FAISS_INDEX_PATH = "faiss_index_bioasq"  # Path all'indice FAISS pre-costruito


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
    # --- Metodo alternativo: Retrieval come Langchain Retriever ---
    # retrieve = vector_store.as_retriever(
    #     search_kwargs={"k": 5}
    # )  # Configura per recuperare i primi 5 chunk più pertinenti
    # relevant_docs = retrieve.get_relevant_documents(query) # Esegue la ricerca
    # print(f"Numero di documenti recuperati (come Retriever): {len(relevant_docs)}")
    # for doc in relevant_docs:
    #     print(f"ID: {doc.metadata['source']}")
    #     print(f"Contenuto: {doc.page_content}")
    #     print("-" * 80)

    # --- Metodo utilizzato: Retrieval per similarità coseno con punteggio ---
    # Converte la query in un embedding e cerca i vettori più simili nell'indice FAISS.
    # Restituisce i documenti (chunk) e il loro punteggio di similarità (distanza coseno).
    # Valori più bassi indicano maggiore similarità.
    retrieve = vector_store.similarity_search_with_score_by_vector(
        embedding=embeddings.embed_query(query),
        k=15,  # Recupera i primi 15 chunk più pertinenti
    )

    print(f"\n--- Chunk Recuperati (Top {len(retrieve)}) ---")
    for doc, score in retrieve:
        print(f"ID Sorgente: {doc.metadata['source']}")
        # print(f"Contenuto: {doc.page_content[:200]}...") # Stampa solo l'inizio per brevità
        print(f"Punteggio Similarità (distanza): {score:.4f}")  # Formatta lo score
        print("-" * 80)

    return retrieve


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

    # --- 3. Recupero dei chunk più simili ---
    print("\nRecupero dei chunk pertinenti dalla query...")
    most_similar_chunks = find_most_similar(
        vector_store=vector_store, embeddings=embeddings, query=query
    )

    # --- 4. Preparazione del contesto per LLM ---
    # Combina il contenuto dei chunk recuperati per formare il contesto
    context_for_llm = "\n".join(
        [chunk.page_content for chunk, _ in most_similar_chunks]
    )

    # --- 5. Generazione della risposta con ollama ---
    # ! SCIRPT FOR OLLAMA MODELs
    # print("\nGenerazione della risposta con Ollama (Mistral)...")
    # try:
    #     response = ollama.chat(
    #         model="mistral",  # Modello di chat da usare (assicurati sia installato: ollama pull mistral)
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": SYSTEM_PROMPT
    #                 + context_for_llm,  # Fornisce il prompt di sistema e il contesto recuperato
    #             },
    #             {
    #                 "role": "user",
    #                 "content": query,  # Fornisce la query originale dell'utente
    #             },
    #         ],
    #     )

    #     # --- 6. Stampa della risposta ---
    #     print("-" * 80)
    #     print("Risposta del modello:")
    #     print(response["message"]["content"])
    #     print("-" * 80)
    # except Exception as e:
    #     print(f"\nErrore durante la chiamata a Ollama: {e}")
    #     print(
    #         "Assicurati che Ollama sia in esecuzione e il modello 'mistral' sia disponibile."
    #     )
    #     print(
    #         "Puoi avviare Ollama con 'ollama serve' e scaricare il modello con 'ollama pull mistral'."
    #     )


if __name__ == "__main__":
    main()
