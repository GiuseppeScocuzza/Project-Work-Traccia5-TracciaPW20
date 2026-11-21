Smart Review Router — Analisi e Smistamento Recensioni Alberghiere
Prototipo in Python con interfaccia web Gradio (gr.Blocks), preprocessing linguistico in italiano con spaCy (it_core_news_sm), generazione dati sintetica tramite Hugging Face (Qwen/Qwen2.5-72B-Instruct) con fallback locale su Faker, e modelli di Machine Learning (scikit-learn) per la classificazione multi-label dei reparti (Housekeeping, Reception, F&B) e l'analisi del sentiment (positivo / negativo).

├── Generatore_gradio.py          # Interfaccia web Gradio, generazione dataset e inferenza (singola / batch CSV)
├── Train_pipeline.py           # Preprocessing spaCy, addestramento ML, valutazione metriche e salvataggio .pkl
├── dataset_recensioni_hotel.csv  # Dataset generato/esportato (creato all'uso)
├── model_dept.pkl                # Classificatore multi-label reparti (OneVsRestClassifier + LogisticRegression)
├── model_sent.pkl                # Classificatore sentiment (LogisticRegression)
├── tfidf.pkl                     # Vettorizzatore TF-IDF (ngram_range=(1, 2))
├── mlb.pkl                       # Codificatore etichette multi-label (MultiLabelBinarizer)
└── README.md                     # Documentazione e istruzioni di avvio

Prerequisiti e Installazione
Clona la repository e spostati nella cartella del programma:

Bash
git clone <URL_DELLA_TUA_REPOSITORY_GITHUB>
cd <NOME_CARTELLA_PROGETTO>

Crea e attiva un ambiente virtuale (consigliato):

Bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

Installa le dipendenze Python:

Bash
pip install gradio pandas scikit-learn spacy faker huggingface_hub

Scarica il modello linguistico italiano di spaCy:

Bash
python -m spacy download it_core_news_sm

Configurazione Token Hugging Face (Opzionale)
Il generatore di recensioni sintetiche prova a usare il modello Qwen/Qwen2.5-72B-Instruct tramite l'API di Hugging Face; se il token non è impostato o se la chiamata API fallisce/va in timeout, il sistema passa automaticamente al fallback locale con Faker.

Linux / macOS:

Bash
export HF_TOKEN="il_tuo_token_huggingface"

Windows (PowerShell):

PowerShell
$env:HF_TOKEN="il_tuo_token_huggingface"

Come Avviare il Programma
Opzione A: Interfaccia Web con Gradio (consigliata)
Avvia la dashboard interattiva:

Bash
python Generatore_gradio.py
Apri nel browser l'indirizzo locale stampato nel terminale (in genere [http://127.0.0.1:7860](http://127.0.0.1:7860)).

All'interno della dashboard troverai tre schede:

Generazione & Training:

Scegli il numero di recensioni da generare (consigliati almeno 50–100 campioni per avere una ripartizione significativa tra training set all'80% e test set al 20%).

Esporta o azzera il file dataset_recensioni_hotel.csv.

Avvia il training per addestrare i modelli, salvare i quattro file .pkl (model_dept.pkl, model_sent.pkl, tfidf.pkl, mlb.pkl) e visualizzare le tabelle delle metriche (Precision, Recall, F1-Score, Support e Accuracy).

Analisi Istantanea:

Inserisci Titolo e Testo di una recensione e clicca su Analizza per ottenere i reparti previsti, la polarità del sentiment e la relativa percentuale di confidenza.

Batch CSV:

Carica un file .csv contenente le recensioni da classificare, visualizza l'anteprima con le predizioni aggiunte e scarica il file CSV elaborato.


Opzione B: Addestramento da Terminale (CLI)
Puoi lanciare direttamente la pipeline di addestramento e valutazione senza aprire l'interfaccia Gradio:

Bash
python Train_pipeline.py

Risoluzione Rapida dei Problemi (Troubleshooting)
it_core_news_sm non trovato: esegui nuovamente python -m spacy download it_core_news_sm all'interno dello stesso ambiente Python in cui avvii gli script.

Avviso "Modelli non trovati" durante l'analisi singola o batch: assicurati di aver prima eseguito il training (dalla prima scheda di Gradio o tramite Train_pipeline.py), verificando che nella cartella siano presenti tutti e quattro i file: model_dept.pkl, model_sent.pkl, tfidf.pkl e mlb.pkl.

Dataset troppo piccolo per la valutazione: se il file CSV ha pochissime righe (es. meno di 5–10 record), il test set al 20% risulterà ridotto e alcune classi o combinazioni multi-label potrebbero non comparire nella valutazione.

Nota sul Dataset e sul Contesto Applicativo
Il progetto è stato sviluppato a scopo dimostrativo e accademico sul caso studio fittizio Grand Hotel Belvedere utilizzando dati sintetici; per un impiego su recensioni reali è consigliata una fase di validazione e calibrazione su dati autentici anonimizzati.