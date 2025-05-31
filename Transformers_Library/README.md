# 📚 Cartella `transformers` – Fine-tuning e Valutazione Modelli Generativi

Questa cartella contiene l’implementazione completa per il **fine-tuning**, la **gestione degli adapter LoRA** e la **valutazione** di modelli generativi per il task di question answering biomedico, basata sulla libreria **Transformers di Hugging Face**.

---

## ⚙️ Scelte Tecnologiche e Motivazioni

Abbiamo scelto Transformers di Hugging Face come ambiente principale per:

- L’ecosistema maturo, costantemente aggiornato, con ampia disponibilità di modelli preaddestrati per NLP e domini specialistici (biomedico).
- Il supporto integrato e trasparente alle tecniche di **parameter-efficient fine-tuning** come LoRA e Adapters, tramite la libreria **PEFT**.
- La facilità di integrazione, personalizzazione e deployment in ambienti locali e cloud.
- Le pipeline automatiche e il vasto supporto di metriche e dataset collegati.

---

## 📁 Struttura della Cartella e Funzioni Principali

| File                                | Funzione                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------- |
| `load_models.py`                    | Caricamento modello e tokenizer, configurazione e avvio del training con parametri PEFT.    |
|                                     | Integrazione LoRA, setup dei parametri di training (learning rate, batch size, ecc.).       |
|                                     | Calcolo delle metriche (ROUGE, BLEU, METEOR) personalizzate tramite `metrics_utils.py`.     |
|                                     | Salvataggio degli adapter e tokenizer fine-tunati.                                          |
| `merge_base_and_adapt_ft_models.py` | Fusione (“merge”) degli adapter LoRA nel modello base per creare un checkpoint stand-alone. |
|                                     | Gestione errori, salvataggio tokenizer e percorsi standardizzati per riutilizzabilità.      |
| `metrics_utils.py`                  | Calcolo delle metriche di generazione (ROUGE, BLEU, METEOR) con pre-elaborazione output.    |

---

## 🧪 Pipeline e Workflow

1. **Preprocessing** e suddivisione dataset (notebook esterno).
2. **Caricamento modello e tokenizer** in `load_models.py`.
3. **Configurazione del training** con parametri ottimizzati per risorse limitate (batch size, accumulo gradienti, precisione).
4. **Addestramento con LoRA** per aggiornare un numero ridotto di parametri.
5. **Valutazione continua** con metriche personalizzate per la qualità della generazione.
6. **Salvataggio** degli adapter LoRA e tokenizer fine-tunati.
7. **Fusione degli adapter** con modello base per produzione o inferenza standalone.

---

## 🛠 Sfide e Soluzioni

- Ottimizzazione dell’uso della memoria per GPU limitate e Apple MPS.
- Gestione delle versioni di `transformers`, `peft` e dipendenze per garantire riproducibilità.
- Salvataggio e caricamento standardizzati per esperimenti ripetibili.
- Bilanciamento tra tempi di training e qualità del modello anche su modelli di dimensioni contenute.

---

## 📋 Requisiti

- Python 3.8+
- Pacchetti:
  ```bash
  pip install -r requirements.txt
  ```

---

## 📌 Note Finali

La cartella **`transformers`** fornisce una pipeline modulare, trasparente e facilmente estendibile per esperimenti avanzati di fine-tuning generativo in ambito biomedico, con particolare attenzione alla **riproducibilità** e **ottimizzazione delle risorse**.

---

[⬆️ Torna al README principale](../README.md)
