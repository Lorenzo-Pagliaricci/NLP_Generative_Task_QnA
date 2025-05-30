# 📁 MLX

Questa cartella contiene tutto il necessario per sperimentare l’uso della libreria **MLX-LM**, sviluppata per l’esecuzione efficiente di modelli linguistici di grandi dimensioni (LLM) su dispositivi **Apple Silicon**. È stata una delle prime soluzioni adottate nel nostro progetto per il fine-tuning e la generazione di risposte da modelli LLM ottimizzati.

---

## 🎯 Obiettivo della cartella

La cartella `MLX` nasce per:

- Testare il fine-tuning di modelli con tecnica **LoRA**, a basso consumo di memoria
- Effettuare **inferenza** e generazione di risposte su prompt personalizzati
- Sfruttare l’hardware Apple per eseguire LLM localmente in modo leggero ed efficiente
- Valutare la qualità delle risposte generate da modelli personalizzati

---

## 📦 Contenuto della cartella

All’interno troverai:

- 📜 **Script `.sh`** per gestire fine-tuning, fusione e inferenza:
  - `fine_tune.sh`: fine-tuning del modello con LoRA
  - `fuse.sh`: fusione degli adapter LoRA nel modello base
  - `generate.sh`: generazione di risposte su prompt di input
  - `chat.sh`, `evaluate.sh`: script non più usati o non funzionanti
- 📄 **Notebook di supporto** (facoltativo)
- 📁 Modelli scaricabili da Hugging Face
- ⚙️ Configurazioni e percorsi dei modelli base e adattati

---

## 🧰 Requisiti

Per eseguire i file di questa cartella:

1. Assicurati di aver installato tutte le dipendenze:

```bash
pip install -r requirements.txt
```

2. Scarica il modello da Hugging Face nel formato compatibile con MLX:

```bash
huggingface-cli download mlx-community/gemma-3-1b-it-4bit \
  --local-dir ./proveIgnazio/models/base/gemma-3-1b-it-4bit
```

3. Esegui gli script .sh direttamente da terminale:

```bash
./fine_tune.sh
./fuse.sh
./generate.sh
```

## 🚀 Perché MLX-LM?

MLX-LM è una libreria progettata per girare in modo ottimale su dispositivi Apple (come MacBook con chip M1/M2/M3), ed è utile quando si vuole:

- Eseguire modelli Hugging Face senza usare GPU dedicate  
- Ridurre l’uso di memoria tramite **quantizzazione**  
- Fare **fine-tuning locale** su modelli leggeri  
- Generare risposte in modalità **offline**

### ✅ Vantaggi principali

- Compatibilità diretta con **Hugging Face**
- Supporto per **LoRA** e tecniche *parameter-efficient*
- Sfrutta appieno la **CPU e GPU Apple Silicon**
- Ideale per **sperimentazioni veloci** e **prototipazione**

---

## ⚠️ Limiti riscontrati

Nonostante il potenziale, durante il progetto abbiamo riscontrato diverse limitazioni nell’uso di MLX-LM:

- La documentazione è ancora **incompleta**
- Mancano strumenti avanzati per modelli **personalizzati o grandi**
- La **community** è piccola, quindi difficile trovare supporto
- Prestazioni **limitate** con modelli più complessi
- Integrazione complicata con tecniche più avanzate (es. **LoRA estese**, **training distribuito**)

---

## 🔄 Perché è stato abbandonato

Abbiamo scelto di non proseguire con MLX-LM come soluzione principale perché:

- La libreria è ancora **immatura**
- Non garantisce **scalabilità** e **flessibilità** necessarie
- Altri framework (come **Transformers + PEFT**) offrono:
  - Supporto migliore  
  - Strumenti più maturi  
  - Community attiva

> 🔍 **Nota:** Nonostante ciò, MLX-LM è stata utile come fase esplorativa e resta una buona opzione per test **rapidi** su hardware Apple.

---

[⬆️ Torna al README principale](../README.md)
