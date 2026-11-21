import os
import pickle
import random
import re
import numpy as np
import pandas as pd
from faker import Faker
import gradio as gr
from huggingface_hub import InferenceClient

from Train_pipeline import preprocess_text, train_and_evaluate

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN) if HF_TOKEN else None
DATASET_PATH = "dataset_recensioni_hotel.csv"

np.random.seed(42)
random.seed(42)
fake = Faker("it_IT")

dataset_store: list[dict] = []


def generate_local_synthetic_fallback(sentiment: str) -> tuple[str, str]:
    piatti_c = [
        "croissant freschi", "caffè espresso", "buffet salato",
        "uova strapazzate", "cappuccino", "frutta di stagione", "pancake"
    ]
    piatti_p = [
        "pasta al forno", "risotto", "grigliata di carne",
        "pesce", "insalata mista", "dessert del giorno", "pizza"
    ]
    orari_m = ["8:00", "9:30", "10:00", "7:30"]
    orari_p = ["19:00", "20:30", "21:00", "18:30"]
    stanze = [str(random.randint(1, 120)) for _ in range(10)]

    if sentiment == "positivo":
        titles = [
            f"Soggiorno perfetto in stanza {random.choice(stanze)}",
            "Ottima esperienza e servizio impeccabile",
            "Ritorneremo sicuramente!",
            "Hotel consigliatissimo per famiglie e coppie",
            "Pienamente soddisfatto del soggiorno",
            f"Un'esperienza indimenticabile grazie a {fake.first_name()}",
            f"Servizio eccellente in camera {random.choice(stanze)}",
            "Accoglienza calorosa e personale sempre disponibile",
            f"Soggiorno rilassante e confortevole in camera {random.choice(stanze)}",
            "Pulizia e ordine impeccabili in tutta la struttura",
        ]
        pools = {
            "Housekeeping": [
                f"La camera {random.choice(stanze)} era impeccabile, profumata e riordinata ogni mattina con cura da {fake.first_name()}.",
                f"Pulizia dei sanitari impeccabile e cambio asciugamani puntuale alle {random.choice(orari_m)}.",
                f"Servizio in camera eccellente: {fake.first_name()} ha lasciato tutto in ordine e pulito.",
                f"La camera {random.choice(stanze)} era sempre perfettamente pulita, con lenzuola fresche e asciugamani morbidi.",
                f"Nonostante un guasto ai sanitari, l'assistenza di {fake.first_name()} è stata rapida e professionale.",
                f"Abbiamo apprezzato il frigo bar in camera {random.choice(stanze)} sempre rifornito e pulito.",
                f"Il letto di camera {random.choice(stanze)} era comodissimo, con lenzuola profumate e cuscini morbidi.",
            ],
            "Reception": [
                f"Check-in rapido gestito con grande professionalità da {fake.first_name()} alla reception.",
                "L'accoglienza al front desk è stata impeccabile e ci hanno fornito ottimi consigli sulla città.",
                f"Il personale della reception, guidato da {fake.first_name()}, è stato estremamente disponibile e cortese.",
                f"Check-out veloce e senza intoppi, con {fake.first_name()} che ci ha assistito con grande efficienza.",
                "I servizi offerti dalla reception sono stati eccellenti, con personale sempre pronto a risolvere ogni esigenza.",
                f"Abbiamo ricevuto suggerimenti preziosi su ristoranti locali da {fake.first_name()} alla reception.",
                "Il personale della reception ha dimostrato grande attenzione ai dettagli durante tutto il soggiorno.",
                f"Nonostante il check-in tardivo alle {random.choice(orari_p)}, {fake.first_name()} ci ha accolto con un sorriso.",
            ],
            "F&B": [
                f"Colazione a buffet ricca e varia, fantastici i {random.choice(piatti_c)} e il {random.choice(piatti_c)}.",
                f"Abbiamo apprezzato la qualità dei piatti serviti a pranzo, con {random.choice(piatti_p)} freschi.",
                f"Il servizio al ristorante è stato impeccabile, con {fake.first_name()} che ci ha consigliato ottimi vini.",
                f"Il ristorante dell'hotel offre piatti squisiti e {fake.first_name()} ci ha serviti con grande garbo.",
                f"Gli orari della colazione erano flessibili e {fake.first_name()} è stato sempre attento e disponibile.",
                f"Abbiamo apprezzato che {fake.first_name()} a colazione abbia preparato opzioni personalizzate per le intolleranze.",
                f"Il servizio al bar è stato eccellente, con {fake.first_name()} che ha preparato cocktail deliziosi.",
                f"Ottima varietà di bevande al bar, con {fake.first_name()} che ci ha consigliato ottimi vini locali.",
            ],
        }
    else:
        titles = [
            "Esperienza deludente sotto diversi aspetti",
            "Pessimo servizio e scarsa attenzione al cliente",
            "Purtroppo non ci siamo proprio",
            "Sconsigliato: rapporto qualità/prezzo pessimo",
            "Molti problemi durante il soggiorno, non torneremo",
            "Servizio inaccettabile e personale poco professionale",
            "Delusione totale",
            f"Esperienza negativa in camera {random.choice(stanze)}",
            "Molti disservizi e poca cura per gli ospiti",
            "Non consiglierei questo hotel a nessuno",
        ]
        pools = {
            "Housekeeping": [
                f"La camera {random.choice(stanze)} era poco pulita, con polvere evidente sui mobili e bagno trascurato.",
                "Pulizia davvero approssimativa: abbiamo trovato capelli altrui nel piatto doccia.",
                f"Il cambio asciugamani è stato saltato per due giorni nella camera {random.choice(stanze)}.",
                "Abbiamo segnalato un problema con il bagno, ma nessuno è venuto a risolverlo.",
                f"La camera {random.choice(stanze)} aveva un odore sgradevole e il materasso usurato.",
                f"La pulizia della camera {random.choice(stanze)} è rimasta insoddisfacente, con lenzuola macchiate.",
                f"Abbiamo trovato ragnatele e polvere sugli scaffali della camera {random.choice(stanze)}.",
                f"Il bagno della camera {random.choice(stanze)} era sporco, con sanitari macchiati e calcare nella doccia.",
                f"La camera {random.choice(stanze)} non era pulita correttamente, con asciugamani sporchi e pavimento appiccicoso.",
            ],
            "Reception": [
                "Attesa infinita al check-in: oltre 40 minuti di coda con un solo addetto alla reception.",
                "Al momento del check-out ci hanno addebitato costi extra non dovuti ed errati.",
                "Il personale della reception è stato scortese e poco disponibile alle nostre richieste.",
                "Abbiamo segnalato un problema con la prenotazione, ma la reception non ha fornito alcuna soluzione.",
                "Durante il check-in il personale della reception ci ha trattato con indifferenza.",
                "Abbiamo riscontrato errori nella prenotazione e la reception non ha mostrato volontà di correggerli.",
                "Il personale della reception non ha fornito informazioni chiare sui servizi dell'hotel.",
                "Ho segnalato un problema con la prenotazione e la reception non ha mostrato volontà di risolverlo.",
                f"Il receptionist {fake.first_name()} è stato scortese e poco professionale durante il check-in.",
                "Abbiamo avuto problemi con la prenotazione online e la reception non ha fornito assistenza adeguata.",
            ],
            "F&B": [
                f"Colazione di scarsa qualità: i {random.choice(piatti_c)} erano freddi e il buffet spoglio alle {random.choice(orari_m)}.",
                "Caffè imbevibile dalle macchinette e servizio ai tavoli estremamente lento.",
                "Il ristorante dell'hotel ha servito piatti mal presentati e poco appetitosi.",
                f"Abbiamo ordinato {random.choice(piatti_p)} ed è stato servito freddo e poco cotto.",
                f"Il servizio al ristorante è stato lento e disorganizzato, con {fake.first_name()} che ha dimenticato l'ordine.",
                f"Abbiamo trovato il cibo servito al ristorante poco fresco, con {random.choice(piatti_p)} sotto le attese.",
                f"Il personale del ristorante era scortese e {fake.first_name()} ha dimenticato le bevande ordinate.",
                f"Problemi con l'ordinazione al ristorante: {fake.first_name()} ha confuso i piatti tra due camere.",
                f"Il servizio al bar è stato lento e {fake.first_name()} ha dimenticato di portare le bevande.",
                "Abbiamo trovato il cibo servito al bar poco fresco e di qualità inferiore.",
            ],
        }

    is_multilabel = random.random() < 0.30
    dept_names = list(pools.keys())

    if is_multilabel:
        selected_depts = sorted(random.sample(dept_names, 2))#Usando sorted() "F&B, Housekeeping" e "Housekeeping, F&B" verranno trattate come due classi uguali nel CSV esportato, allineando l'etichetta del dataset all'ordine alfabetico usato da MultiLabelBinarizer
        part1 = random.choice(pools[selected_depts[0]])
        part2 = random.choice(pools[selected_depts[1]])
        body = f"{part1} Inoltre, {part2[0].lower() + part2[1:]}"
        dept_label = f"{selected_depts[0]}, {selected_depts[1]}"
    else:
        selected_dept = random.choice(dept_names)
        body = random.choice(pools[selected_dept])
        dept_label = selected_dept

    title = random.choice(titles)
    return f"TITOLO: {title}\nTESTO: {body}", dept_label


def generate_review_with_llm(sentiment_target: str) -> tuple[str | None, str | None]:
    if client is None:
        return None, None
    else:
    
        prompt = (
            f"Scrivi una breve recensione d'hotel in italiano con sentiment {sentiment_target.upper()}.\n"
            "Puoi citare uno o due reparti tra Housekeeping, Reception e F&B.\n"
            "Rispondi ESCLUSIVAMENTE in questo formato:\n"
            "TITOLO: <titolo breve>\n"
            "REPARTI: <uno o due nomi separati da virgola tra Housekeeping, Reception, F&B>\n"
            "TESTO: <testo della recensione>"
        )
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Sei un generatore di recensioni sintetiche per dataset NLP."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=160,
                temperature=0.9,
            )
            raw = response.choices[0].message.content or ""
            t_match = re.search(r"TITOLO:\s*(.*)", raw, re.IGNORECASE)
            d_match = re.search(r"REPARTI:\s*(.*)", raw, re.IGNORECASE)
            b_match = re.search(r"TESTO:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
            if t_match and b_match:
                valid = {"Housekeeping", "Reception", "F&B"}
                parsed_depts = [
                    d.strip()
                    for d in (d_match.group(1) if d_match else "").split(",")
                    if d.strip() in valid
                ]
                dept_str = ", ".join(parsed_depts) if parsed_depts else "Reception"
                return f"TITOLO: {t_match.group(1).strip()}\nTESTO: {b_match.group(1).strip()}", dept_str
        except Exception:
            pass
        return None, None


def generate_batch(num_samples: int):
    global dataset_store
    sentiments = ["positivo", "negativo"] * (num_samples // 2)
    if num_samples % 2 != 0:
        sentiments.append(random.choice(["positivo", "negativo"]))
    random.shuffle(sentiments)

    new_entries = []
    for sentiment in sentiments:
        raw_output, dept_label = generate_review_with_llm(sentiment)
        if raw_output is None or dept_label is None:
            raw_output, dept_label = generate_local_synthetic_fallback(sentiment)

        t_match = re.search(r"TITOLO:\s*(.*)", raw_output, re.IGNORECASE)
        b_match = re.search(r"TESTO:\s*(.*)", raw_output, re.IGNORECASE | re.DOTALL)
        title = t_match.group(1).strip() if t_match else "Soggiorno in Hotel"
        body = b_match.group(1).strip() if b_match else raw_output.strip()

        new_entries.append({
            "id": f"REV_{len(dataset_store) + len(new_entries) + 1:04d}",
            "title": title,
            "body": body,
            "department": dept_label,
            "sentiment": sentiment,
        })

    dataset_store.extend(new_entries)
    df = pd.DataFrame(dataset_store)
    return df, f"Generati {len(new_entries)} campioni. Totale in memoria: {len(dataset_store)}."


def export_csv():
    if not dataset_store:
        return None, "⚠️ Nessun dato da esportare!"
    df = pd.DataFrame(dataset_store, columns=["id", "title", "body", "department", "sentiment"])
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")
    return DATASET_PATH, f"✅ Dataset esportato in {DATASET_PATH}."


def clear_dataset():
    global dataset_store
    dataset_store = []
    empty_df = pd.DataFrame(columns=["id", "title", "body", "department", "sentiment"])
    return empty_df, "Dataset azzerato."


def run_training_from_ui():
    empty_df = pd.DataFrame(columns=["Classe / Metrica", "precision", "recall", "f1-score", "support"])
    if dataset_store:
        export_csv()
    elif not os.path.exists(DATASET_PATH):
        return (
            "⚠️ Genera prima un dataset o salva 'dataset_recensioni_hotel.csv'.",
            empty_df,
            empty_df,
        )
    summary, df_dept, df_sent = train_and_evaluate(DATASET_PATH)
    return summary, df_dept, df_sent


def load_ml_assets():
    files = ["model_dept.pkl", "model_sent.pkl", "tfidf.pkl", "mlb.pkl"]
    if not all(os.path.exists(f) for f in files):
        return None
    try:
        with open("model_dept.pkl", "rb") as f:
            m_dept = pickle.load(f)
        with open("model_sent.pkl", "rb") as f:
            m_sent = pickle.load(f)
        with open("tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)
        with open("mlb.pkl", "rb") as f:
            mlb = pickle.load(f)
        return tfidf, m_dept, m_sent, mlb
    except Exception:
        return None


def analyze_single_review(title: str, body: str):
    full_text = f"{title or ''} {body or ''}".strip()
    if not full_text:
        return "Inserisci titolo o testo", "-", "0.0%"

    assets = load_ml_assets()
    if assets is None:
        return "Modelli non trovati: avvia prima il training", "-", "0.0%"

    tfidf, m_dept, m_sent, mlb = assets
    try:
        cleaned = preprocess_text(full_text)
        vec = tfidf.transform([cleaned])

        dept_bin = m_dept.predict(vec)
        active_depts = mlb.inverse_transform(dept_bin)[0]
        dept_str = ", ".join(active_depts) if active_depts else "Generale"

        sent_pred = str(m_sent.predict(vec)[0])
        confidence = float(max(m_sent.predict_proba(vec)[0]))
        sent_label = "🟢 POSITIVO" if sent_pred.lower() == "positivo" else "🔴 NEGATIVO"
        return dept_str, sent_label, f"{confidence * 100:.1f}%"
    except Exception as exc:
        return f"Errore predizione: {exc}", "-", "0.0%"


def process_batch_csv(file_obj):
    empty_preview = pd.DataFrame()
    if file_obj is None:
        return None, "⚠️ Carica un file CSV.", empty_preview

    assets = load_ml_assets()
    if assets is None:
        return None, "⚠️ Modelli non trovati: esegui prima il training.", empty_preview

    tfidf, m_dept, m_sent, mlb = assets
    try:
        batch_df = pd.read_csv(file_obj.name)
        if not {"title", "body"}.issubset(batch_df.columns):
            return None, "⚠️ Il CSV deve contenere le colonne 'title' e 'body'.", empty_preview

        cleaned = (batch_df["title"].fillna("") + " " + batch_df["body"].fillna("")).apply(preprocess_text)
        vecs = tfidf.transform(cleaned)

        dept_bin = m_dept.predict(vecs)
        batch_df["predicted_department"] = [
            ", ".join(row) if row else "Generale" for row in mlb.inverse_transform(dept_bin)
        ]
        batch_df["predicted_sentiment"] = m_sent.predict(vecs)

        out_path = "batch_predictions_output.csv"
        batch_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path, "✅ Predizioni batch completate.", batch_df
    except Exception as exc:
        return None, f"❌ Errore batch: {exc}", empty_preview


with gr.Blocks(title="Hotel Feedback Router") as demo:
    gr.Markdown("# 🏨 Smart Review Router & ML Pipeline Integration")

    with gr.Tabs():
        with gr.TabItem("📊 1. Generazione & Training"):
            with gr.Row():
                with gr.Column(scale=1):
                    num_samples = gr.Slider(50, 500, value=50, step=2, label="Recensioni da generare")
                    btn_generate = gr.Button("Genera Dataset Sintetico", variant="primary")
                    btn_train = gr.Button("🚀 Avvia Train Pipeline & Valutazione", variant="secondary")
                    btn_export = gr.Button("Esporta CSV")
                    btn_clear = gr.Button("Azzera Dataset", variant="stop")
                    file_out = gr.File(label="Scarica CSV")
                    status_box = gr.Textbox(label="Stato Operazione", interactive=False)

                
 
                with gr.Column(scale=2):
                    gr.Markdown("### Anteprima Dataset")
                    table_view = gr.Dataframe(headers=["id", "title", "body", "department", "sentiment"])
                    gr.Markdown("### Valutazione Reparto (Multi-Label)")
                    dept_report_table = gr.Dataframe(
                        headers=["Classe / Metrica", "precision", "recall", "f1-score", "support"],
                        interactive=False,
                    )

                    gr.Markdown("### Valutazione Sentiment")
                    sent_report_table = gr.Dataframe(
                        headers=["Classe / Metrica", "precision", "recall", "f1-score", "support"],
                        interactive=False,
                    )


            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Note:")
                    gr.Markdown(
                        "- La generazione di recensioni è basata su un modello LLM (Qwen2.5-72B-Instruct) con fallback locale.\n"
                        "- Il training dei modelli ML utilizza Logistic Regression con TF-IDF.\n"
                        "- I reparti possono essere multi-label (es. 'Housekeeping, Reception').\n"
                        "- Assicurati di avere almeno 50-100 recensioni per un training significativo."
                )
            
                with gr.Column(scale=1):
                    gr.Markdown("### Disclaimer:")
                    gr.Markdown(
                        "- Questo strumento è a scopo dimostrativo e non garantisce accuratezza in contesti reali.\n"
                        "- I dati generati sono sintetici e non rappresentano recensioni reali.\n"
                        "- L'uso dei modelli LLM può comportare costi se si utilizzano API esterne."
                    )
            
                with gr.Column(scale=1):
                    gr.Markdown("### Suggerimenti per l'Uso:")
                    gr.Markdown(
                        "- Per generare un dataset bilanciato, seleziona un numero pari di recensioni.\n"
                        "- Dopo la generazione, esegui il training per ottenere i modelli ML.\n"
                        "- Puoi analizzare singole recensioni o caricare batch CSV per predizioni multiple."
                    )

        with gr.TabItem("✍️ 2. Analisi Istantanea"):
            with gr.Row():
                with gr.Column():
                    in_title = gr.Textbox(label="Titolo", placeholder="Es. Pulizia e check-in")
                    in_body = gr.Textbox(
                        label="Testo",
                        lines=4,
                        placeholder="Es. Camera pulitissima ma attesa lunga alla reception.",
                    )
                    btn_analyze = gr.Button("Analizza", variant="primary")
                with gr.Column():
                    out_dept = gr.Textbox(label="Reparto", interactive=False)
                    out_sent = gr.Textbox(label="Sentiment", interactive=False)
                    out_conf = gr.Textbox(label="Confidenza", interactive=False)

        with gr.TabItem("📁 3. Batch CSV"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label="Carica CSV (colonne: title, body)", file_types=[".csv"])
                    btn_batch = gr.Button("Esegui Predizioni", variant="primary")
                    batch_status = gr.Textbox(label="Stato Batch", interactive=False)
                    file_dl = gr.File(label="Scarica Risultati")

                with gr.Column(scale=2):
                    gr.Markdown("### Anteprima Predizioni Batch")
                    batch_preview = gr.Dataframe(interactive=False)

    btn_generate.click(fn=generate_batch, inputs=[num_samples], outputs=[table_view, status_box])
    btn_export.click(fn=export_csv, inputs=[], outputs=[file_out, status_box])
    btn_clear.click(fn=clear_dataset, inputs=[], outputs=[table_view, status_box])
    btn_train.click(fn=run_training_from_ui,inputs=[],outputs=[status_box, dept_report_table, sent_report_table])
    btn_analyze.click(fn=analyze_single_review, inputs=[in_title, in_body], outputs=[out_dept, out_sent, out_conf])
    btn_batch.click(fn=process_batch_csv,inputs=[file_in],outputs=[file_dl, batch_status, batch_preview],)

if __name__ == "__main__":
    demo.launch()