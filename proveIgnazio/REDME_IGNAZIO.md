Questa è la pipeline per capire come eseguire il codice di Ignazio.


### 0. Prerequisiti
1. Ricordti di installare le librerie necessarie per eseguire il codice, trovi tutto nel file requirements.txt.
usa il comando
```bash
pip install -r requirements.txt
```

2. Il file 'DataSplitting.ipynb' serve per dividere i dati in train, validation e test.

3. Il file 'run_eval.ipynb' serve per eseguire la valutazione del modello una volta fine tunato e fuso. <br>
Questo script restituirà un file csv contenente le domande e le rispste generate dal modello di base e quello fine tunato.
Inoltre conterà anche le metriche di performance associate alle risposte generate rispetto a quelle aspettate.


4. Per eseguire i file .sh basta andare sul terminale e scrivere il comando
```bash
./percorso_del_file/nome_del_file.sh
```


### 1. fine_tune.sh
Quesro script serve per eseguire il fine tuning del modello.
Per scaricare un modello da fine tunare prima bisogna scaricarlo 

```bash
huggingface-cli download mlx-community/gemma-3-1b-it-4bit  --local-dir ./proveIgnazio/models/base/gemma-3-1b-it-4bit
```

Questo comando, da eseguire su terminale, scarica il modello gemma-3-1b-it-4bit in nellla cartella specificata in --local-dir.

Successivamente c'è lo script vero e propio per il fine tuning.

### 2. fuse.sh
Questo script serve per eseguire fondere i pesi del modello fine tunato con quelli del modello base.
Per eseguire questo script è necessario avere il modello fine tunato e il modello base.

### 3. chat.sh
Quesro script serve per eseguire il modello fine tunato. 
Permette banalmente di chattare con il modello fine tunato.


### 4. note 
I file di 'evaluate.sh' e 'generate.sh' per ora non funzionano/sono inutili

