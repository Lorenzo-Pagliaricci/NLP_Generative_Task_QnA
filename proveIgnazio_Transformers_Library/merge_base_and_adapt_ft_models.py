from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import dotenv_values
import os
import torch

'''
Caricamento delle variabili d'ambiente dal file .env.
Verifica la presenza del file e delle chiavi necessarie per i percorsi e i nomi dei modelli.
'''
config_env = dotenv_values(".env")
if not config_env:
    print("Errore: il file .env non è stato trovato o è vuoto.")
    print(
        "Assicurati che il file .env sia nella stessa directory dello script o fornisci il percorso corretto."
    )
    exit()

# Nome del modello base come specificato in .env e usato in load_models.py
BASE_MODEL_HF_NAME = config_env.get("BYT5_SMALL_300M")
if not BASE_MODEL_HF_NAME:
    print("Errore: la variabile T5_SMALL_60M non è definita nel file .env.")
    exit()

# Suffisso del modello usato per salvare gli adapter in load_models.py
ADAPTER_MODEL_SUFFIX = "BYT5_SMALL_300M"

SAVED_MODEL_BASE_PATH = config_env.get("SAVED_MODEL_PATH")
if not SAVED_MODEL_BASE_PATH:
    print("Errore: la variabile SAVED_MODEL_PATH non è definita nel file .env.")
    exit()

ADAPTER_MODEL_PATH = f"{SAVED_MODEL_BASE_PATH}_{ADAPTER_MODEL_SUFFIX}" + "_V2"
TOKENIZER_SAVED_PATH = f"{SAVED_MODEL_BASE_PATH}_{ADAPTER_MODEL_SUFFIX}_Tokenizer" + "_V2"
MERGED_MODEL_SAVE_PATH = os.path.join(
    SAVED_MODEL_BASE_PATH, f"merged_{ADAPTER_MODEL_SUFFIX}_for_rag_V2"
)

#Stampa riepilogo dei percorsi e delle impostazioni usate per la fusione del modello.

print(f"--- Inizio del processo di unione del modello LoRA ---")
print(f"Modello base da Hugging Face: {BASE_MODEL_HF_NAME}")
print(f"Percorso adapter LoRA: {ADAPTER_MODEL_PATH}")
print(f"Percorso tokenizer salvato: {TOKENIZER_SAVED_PATH}")
print(f"Percorso di salvataggio del modello unito: {MERGED_MODEL_SAVE_PATH}")

os.makedirs(MERGED_MODEL_SAVE_PATH, exist_ok=True)

#1. Caricamento del modello base da Hugging Face, con ottimizzazioni di memoria.
print(f"\n1. Caricamento del modello base: {BASE_MODEL_HF_NAME}...")
try:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_HF_NAME,
        torch_dtype=torch.float16,  # Usa float16 per compatibilità e dimensioni ridotte
        low_cpu_mem_usage=True,
    )
    print("Modello base caricato.")
except Exception as e:
    print(f"Errore durante il caricamento del modello base: {e}")
    exit()

#2. Caricamento degli adapter LoRA e applicazione al modello base.
print(f"\n2. Caricamento degli adapter LoRA da: {ADAPTER_MODEL_PATH}...")
try:
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)
    print("Adapter LoRA caricati e applicati al modello base.")
except Exception as e:
    print(f"Errore durante il caricamento degli adapter LoRA: {e}")
    print(
        f"Assicurati che il percorso '{ADAPTER_MODEL_PATH}' contenga gli adapter corretti (es. adapter_model.bin)."
    )
    exit()

#3. Fusione dei pesi LoRA nel modello base, rendendolo indipendente dagli adapter.
print("\n3. Unione dei pesi LoRA nel modello base...")
try:
    merged_model = peft_model.merge_and_unload()
    print("Unione completata.")
except Exception as e:
    print(f"Errore durante l'unione dei pesi: {e}")
    exit()


#4. Salvataggio del modello unito in un percorso locale specificato.
print(f"\n4. Salvataggio del modello unito in: {MERGED_MODEL_SAVE_PATH}...")
try:
    merged_model.save_pretrained(MERGED_MODEL_SAVE_PATH)
    print("Modello unito salvato.")
except Exception as e:
    print(f"Errore durante il salvataggio del modello unito: {e}")
    exit()


#5. Caricamento del tokenizer associato al modello, con fallback al modello base in caso di errore, e salvataggio nella stessa directory del modello unito.
print(f"\n5. Caricamento e salvataggio del tokenizer...")
try:
    print(f"Tentativo di caricare il tokenizer da: {TOKENIZER_SAVED_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SAVED_PATH)
    print("Tokenizer caricato dal percorso salvato.")
except Exception as e_saved:
    print(
        f"Non è stato possibile caricare il tokenizer da '{TOKENIZER_SAVED_PATH}'. Errore: {e_saved}"
    )
    print(
        f"Tentativo di caricare il tokenizer dal modello base Hugging Face: {BASE_MODEL_HF_NAME}"
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_HF_NAME)
        print("Tokenizer caricato dal modello base Hugging Face.")
    except Exception as e_base:
        print(f"Errore durante il caricamento del tokenizer dal modello base: {e_base}")
        print("Salvataggio del tokenizer fallito.")
        exit()

try:
    tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH)
    print(f"Tokenizer salvato in: {MERGED_MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Errore durante il salvataggio del tokenizer: {e}")
    exit()

#Conclusione del processo. Il modello unito e il tokenizer sono pronti per essere utilizzati,
#ad esempio in un Modelfile per Ollama.
print(f"\n--- Processo completato ---")
print(f"Il modello unito e il tokenizer sono pronti in: {MERGED_MODEL_SAVE_PATH}")
print("Ora puoi creare un Modelfile per Ollama che punti a questa directory.")
