# 📚 Documentazione delle Cartelle del Progetto

In questo lavoro presentiamo un sistema di **domanda-risposta generativo (RAG)** specificamente progettato per il dominio biomedico. L’obiettivo è superare i limiti di efficienza e generalizzazione tipici dei modelli di grandi dimensioni, adottando modelli leggeri e strategie di fine-tuning parameter-efficient (LoRA). La pipeline integra un retrieval semantico basato su FAISS e modelli generativi, addestrati con tecniche PEFT tramite la libreria Transformers di Hugging Face.

Abbiamo privilegiato dataset specializzati come **rag-mini-bioasq** per la sperimentazione, focalizzandoci su un sistema modulare, efficiente e scalabile anche su hardware limitato come Apple Silicon. La valutazione avviene tramite metriche automatizzate (ROUGE, BLEU, METEOR, Recall@k) per garantire risposte precise e contestualizzate. Il codice sorgente è disponibile su GitHub, organizzato in moduli per semplificare lo sviluppo, la sperimentazione e la riproducibilità.

---

## 📂 Cartelle e relative descrizioni

- ### 📁 [Prove_passate](Prove_passate/README.md)

  **Archivio storico di esperimenti preliminari** e test iniziali su tecniche di NLP e retrieval, inclusi tentativi con modelli e framework vari. Qui si trovano script sperimentali, note e prove che hanno guidato la definizione della pipeline attuale, offrendo un contesto di sviluppo e confronto.

- ### 📁 [MLX_LM](MLX_LM/README.md)

  Contiene l’implementazione iniziale basata su **Apple MLX**, ambiente sperimentale per il fine-tuning e l’inferenza di modelli linguistici. Questa cartella documenta la fase di prototipazione e i limiti incontrati, giustificando la successiva migrazione verso la libreria Transformers di Hugging Face per una soluzione più robusta e modulare.

- ### 📁 [Transformers_Library](Transformers_Library/README.md)

  Implementa la pipeline avanzata e ottimizzata di fine-tuning generativo basata su **Hugging Face Transformers**. Qui si trovano gli script principali per il caricamento, l’addestramento con tecniche PEFT (LoRA), la valutazione automatica tramite metriche dedicate e la gestione modulare dei modelli. Questa cartella rappresenta il cuore del progetto, combinando efficienza, riproducibilità e flessibilità d’uso.

- ### 📁 [RAG_system](RAG_system/README.md)

  Implementa il sistema completo di **Retrieval-Augmented Generation (RAG)** specifico per il dominio biomedico. Include il modulo di retrieval con FAISS per l’estrazione dinamica dei contesti rilevanti e la generazione delle risposte con modelli generativi fine-tunati tramite LoRA. Questa cartella contiene gli script chiave per la costruzione dell’indice, l’inferenza e la valutazione, rappresentando il cuore operativo del progetto.
