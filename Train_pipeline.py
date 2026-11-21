import pickle
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

try:
    nlp = spacy.load("it_core_news_sm", disable=["parser", "ner"])
except OSError:
    print("⚠️ Modello spaCy non trovato. Esegui: python -m spacy download it_core_news_sm")
    raise


def preprocess_text(text: str) -> str:
    """Lowercase, rimozione stopword/punteggiatura/numeri e lemmatizzazione con spaCy."""
    doc = nlp(str(text).lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and not token.like_num
    ]
    return " ".join(tokens)


def _report_to_dataframe(y_true, y_pred) -> pd.DataFrame:
    """Converte il classification_report in un DataFrame pulito e arrotondato."""
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df_rep = pd.DataFrame(report_dict).transpose().reset_index()
    df_rep.rename(columns={"index": "Classe / Metrica"}, inplace=True)

    # Arrotonda le colonne numeriche e formatta il support come intero
    for col in ["precision", "recall", "f1-score"]:
        if col in df_rep.columns:
            df_rep[col] = df_rep[col].round(2)
    if "support" in df_rep.columns:
        df_rep["support"] = df_rep["support"].astype(int)

    return df_rep


def train_and_evaluate(csv_path: str = "dataset_recensioni_hotel.csv"):
    """Addestra i modelli, salva i .pkl e restituisce (summary_text, df_dept, df_sent)."""
    empty_df = pd.DataFrame(columns=["Classe / Metrica", "precision", "recall", "f1-score", "support"])

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return f"❌ Dataset '{csv_path}' non trovato!", empty_df, empty_df

    required_cols = {"title", "body", "department", "sentiment"}
    if not required_cols.issubset(df.columns) or len(df) < 5:
        return "⚠️ Dataset non valido o con meno di 5 righe.", empty_df, empty_df

    df["full_text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).apply(preprocess_text)
    # Estrazione e codifica Multi-Label dei reparti
    y_dept_lists = (
        df["department"]
        .fillna("Generale")
        .apply(lambda x: [d.strip() for d in str(x).split(",") if d.strip()])
    )
    mlb = MultiLabelBinarizer()
    y_dept = mlb.fit_transform(y_dept_lists)
    y_sent = df["sentiment"].fillna("negativo")

    if y_dept.shape[1] == 0 or len(y_sent.unique()) < 2:
        return "⚠️ Servono almeno 2 classi diverse per reparto e sentiment.", empty_df, empty_df
    # Divisione Train/Test (80% training, 20% test)
    x_train, x_test, y_dept_train, y_dept_test, y_sent_train, y_sent_test = train_test_split(
        df["full_text"], y_dept, y_sent, test_size=0.20, random_state=42
    )
    # Vettorizzazione e addestramento modelli
    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    x_train_vec = tfidf.fit_transform(x_train)
    x_test_vec = tfidf.transform(x_test)

    model_dept = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000)).fit(
        x_train_vec, y_dept_train
    )
    model_sent = LogisticRegression(C=1.0, max_iter=1000).fit(x_train_vec, y_sent_train)

    # Predizioni e conversione etichette
    dept_pred_bin = model_dept.predict(x_test_vec)
    y_dept_test_str = [", ".join(row) if row else "Generale" for row in mlb.inverse_transform(y_dept_test)]
    dept_pred_str = [", ".join(row) if row else "Generale" for row in mlb.inverse_transform(dept_pred_bin)]
    sent_pred = model_sent.predict(x_test_vec)

    acc_dept = accuracy_score(y_dept_test_str, dept_pred_str)
    acc_sent = accuracy_score(y_sent_test, sent_pred)

    df_dept_report = _report_to_dataframe(y_dept_test_str, dept_pred_str)
    df_sent_report = _report_to_dataframe(y_sent_test, sent_pred)

    with open("model_dept.pkl", "wb") as f:
        pickle.dump(model_dept, f)
    with open("model_sent.pkl", "wb") as f:
        pickle.dump(model_sent, f)
    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("mlb.pkl", "wb") as f:
        pickle.dump(mlb, f)

    summary = (
        f"✅ Modelli e vettorizzatore salvati (.pkl) | "
        f"Accuracy Reparto: {acc_dept:.4f} | "
        f"Accuracy Sentiment: {acc_sent:.4f}"
    )
    return summary, df_dept_report, df_sent_report


if __name__ == "__main__":
    msg, d_rep, s_rep = train_and_evaluate()
    print(msg)
    print("\n--- REPARTI ---\n", d_rep)
    print("\n--- SENTIMENT ---\n", s_rep)