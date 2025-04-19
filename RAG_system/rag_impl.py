from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama


FAISS_INDEX_PATH = "faiss_index_bioasq"


def load_vector_store(embeddings):
    # --- Caricamento del database FAISS ---
    print(f"Caricamento dell'indice FAISS da '{FAISS_INDEX_PATH}'...")
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )  # Aggiunto allow_dangerous_deserialization per compatibilità
    print("Indice FAISS caricato.")

    return vector_store


def find_most_similar(vector_store, embeddings, query):
    # --- Retrieval as retriver ---
    # retrieve = vector_store.as_retriever(
    #     search_kwargs={"k": 5}
    # )  # Recupera i primi 5 chunk più pertinenti

    # query = "What is the implication of histone lysine methylation in medulloblastoma?"

    # relevant_docs = retrieve.get_relevant_documents(query)
    # print(f"Numero di documenti recuperati: {len(relevant_docs)}")
    # for doc in relevant_docs:
    #     print(f"ID: {doc.metadata['source']}")
    #     print(f"Contenuto: {doc.page_content}")
    #     print("-" * 80)

    # --- Retrieval as cosine similarity with score by vector ---
    retrieve = vector_store.similarity_search_with_score_by_vector(
        embedding=embeddings.embed_query(query),
        k=15,  # Recupera i primi 15 chunk più pertinenti
    )

    print(f"Numero di chunk recuperati: {len(retrieve)}")
    for doc, score in retrieve:
        print(f"ID: {doc.metadata['source']}")
        print(f"Contenuto: {doc.page_content}")
        print(f"Score: {score}")
        print("-" * 80)

    return retrieve


def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """

    query = "What is the implication of histone lysine methylation in medulloblastoma?"

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )  # Devi usare lo stesso modello di embedding

    vector_store = load_vector_store(embeddings=embeddings)

    # --- Recupero dei chunk più simili ---
    most_similar_chunks = find_most_similar(
        vector_store=vector_store, embeddings=embeddings, query=query
    )

    # --- Generazione della risposta con ollama ---
    response = ollama.chat(
        model="mistral",  # Modello di chat da usare
        # ollama pull mistral -> per il download
        # ollama rm mistral -> per rimuovere il modello
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join([chunk.page_content for chunk, _ in most_similar_chunks]),
            },
            {
                "role": "user",
                "content": query,
            },
        ],
    )

    # Stampa la risposta
    print("-" * 80)
    print("Risposta del modello:")
    print(response["message"]["content"])
    print("-" * 80)


if __name__ == "__main__":
    main()
