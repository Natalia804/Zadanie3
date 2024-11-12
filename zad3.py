import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load data function with thousands separator handling
@st.cache_data
def load_data(file, sep=";"):
    return pd.read_csv(file, sep=sep, thousands=',')
    
st.header("Uczenie maszynowe - Zadanie 3")
st.subheader("Zuzanna Deszcz 413481, Natalia Łyś 412728")

# Wczytanie danych
file_option = st.selectbox("Wybierz plik:", ["zad3_Airline.csv", "zad3_Stroke.csv"])
data = load_data(file_option)
st.write("Podgląd danych:", data.head())

# Liczba brakujących wartości
st.write("Liczba brakujących wartości w każdej kolumnie:")
st.write(data.isnull().sum())

# Ustalanie kolumn kategorycznych i numerycznych na podstawie wybranego pliku
if file_option == "zad3_Stroke.csv":
    target_column = 'stroke'
    numeric_cols = ["age", "avg_glucose_level", "bmi"]
    categorical_cols = ["gender", "hypertension", "heart_disease", "ever_married",
                        "work_type", "Residence_type", "smoking_status", "stroke"]
elif file_option == "zad3_Airline.csv":
    target_column = 'satisfaction'
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if "Gate.location" in data.columns:
        categorical_cols.append("Gate.location")
        if "Gate.location" in numeric_cols:
            numeric_cols.remove("Gate.location")

# Identyfikacja zmiennych z brakami danych
missing_numeric_cols = [col for col in numeric_cols if data[col].isnull().any()]
missing_categorical_cols = [col for col in categorical_cols if data[col].isnull().any()]

# Identyfikacja zmiennych objaśniających
numeric_explanatory_cols = [col for col in numeric_cols if col != target_column]
categorical_explanatory_cols = [col for col in categorical_cols if col != target_column]

# Tworzenie list zmiennych do wyświetlenia w zakładkach (tylko zmienne objaśniające z brakami danych)
numeric_columns_to_include = [col for col in numeric_explanatory_cols if col in missing_numeric_cols]
categorical_columns_to_include = [col for col in categorical_explanatory_cols if col in missing_categorical_cols]

# Tworzenie zakładek dla zmiennych numerycznych, kategorycznych i modeli ML
tab1, tab2, tab3 = st.tabs(["Zmiennie numeryczne", "Zmiennie kategoryczne", "Modelowanie"])

# Zakładka 1: Analiza zmiennych numerycznych
with tab1:
    st.write("## Analiza zmiennych numerycznych")

    # Sprawdzenie, czy są zmienne do wyświetlenia
    if not numeric_columns_to_include:
        st.write("Brak zmiennych numerycznych objaśniających z brakami danych.")
    else:
        # **Opis metod eliminacji braków danych:**
        st.markdown("""
        a) Usunięcie wszystkich wierszy z brakującymi wartościami.

        b) Wypełnienie braków w zmiennych numerycznych średnią wartością z kolumny.

        c) Wypełnienie braków w zmiennych numerycznych medianą wartością z kolumny.

        d) Wypełnienie braków losowymi wartościami z dostępnych w kolumnie.
        """)

        # Funkcja do obliczania statystyk
        def calculate_stats(df):
            return pd.DataFrame({
                'Średnia': df[numeric_columns_to_include].mean(),
                'Odchylenie standardowe': df[numeric_columns_to_include].std()
            })

        # Oryginalne statystyki (z brakami)
        stats_original = calculate_stats(data)

        # Usunięcie wierszy z brakami
        data_dropped = data.dropna()
        stats_dropped = calculate_stats(data_dropped)

        # Uzupełnienie średnią (dla zmiennych numerycznych)
        imputer_mean = SimpleImputer(strategy="mean")
        data_mean = data.copy()
        data_mean[numeric_cols] = imputer_mean.fit_transform(data[numeric_cols])
        # Uzupełnienie modą (dla zmiennych kategorycznych)
        imputer_mode = SimpleImputer(strategy="most_frequent")
        data_mean[categorical_cols] = imputer_mode.fit_transform(data[categorical_cols])
        stats_mean = calculate_stats(data_mean)

        # Uzupełnienie medianą
        imputer_median = SimpleImputer(strategy="median")
        data_median = data.copy()
        data_median[numeric_cols] = imputer_median.fit_transform(data[numeric_cols])
        data_median[categorical_cols] = imputer_mode.fit_transform(data[categorical_cols])
        stats_median = calculate_stats(data_median)

        # Uzupełnienie losowymi wartościami
        data_random = data.copy()
        for col in numeric_cols:
            data_random[col] = data_random[col].apply(
                lambda x: x if pd.notnull(x) else np.random.choice(data[col].dropna()))
        for col in categorical_cols:
            data_random[col] = data_random[col].apply(
                lambda x: x if pd.notnull(x) else np.random.choice(data[col].dropna()))
        stats_random = calculate_stats(data_random)

        # Tworzenie DataFrame ze statystykami
        stats_combined = pd.DataFrame({
            "Oryginalne (z brakami)": stats_original['Średnia'],
            "Usunięcie wierszy": stats_dropped['Średnia'],
            "Uzupełnienie średnią": stats_mean['Średnia'],
            "Uzupełnienie medianą": stats_median['Średnia'],
            "Uzupełnienie losowymi wartościami": stats_random['Średnia']
        }).T

        stats_combined_std = pd.DataFrame({
            "Oryginalne (z brakami)": stats_original['Odchylenie standardowe'],
            "Usunięcie wierszy": stats_dropped['Odchylenie standardowe'],
            "Uzupełnienie średnią": stats_mean['Odchylenie standardowe'],
            "Uzupełnienie medianą": stats_median['Odchylenie standardowe'],
            "Uzupełnienie losowymi wartościami": stats_random['Odchylenie standardowe']
        }).T

        # Wybór statystyki i kolumny
        stat_option = st.selectbox("Wybierz typ statystyki do wyświetlenia:", ["Średnia", "Odchylenie standardowe"])
        numeric_column_option = st.selectbox("Wybierz kolumnę numeryczną do analizy:", numeric_columns_to_include)

        # Wizualizacja wyników
        if numeric_column_option in stats_combined.columns:
            if stat_option == "Średnia":
                st.write(f"### Porównanie średnich dla kolumny '{numeric_column_option}'")
                fig, ax = plt.subplots(figsize=(10, 6))
                stats_combined[numeric_column_option].plot(kind="bar", ax=ax)
                ax.set_ylabel("Średnia")
                ax.set_xlabel("Metoda")
                ax.set_title(f"Porównanie średnich dla kolumny '{numeric_column_option}'")
                st.pyplot(fig)
            else:
                st.write(f"### Porównanie odchylenia standardowego dla kolumny '{numeric_column_option}'")
                fig, ax = plt.subplots(figsize=(10, 6))
                stats_combined_std[numeric_column_option].plot(kind="bar", ax=ax)
                ax.set_ylabel("Odchylenie standardowe")
                ax.set_xlabel("Metoda")
                ax.set_title(f"Porównanie odchylenia standardowego dla kolumny '{numeric_column_option}'")
                st.pyplot(fig)

# Zakładka 2: Analiza zmiennych kategorycznych
with tab2:
    st.write("## Analiza zmiennych kategorycznych")

    # Sprawdzenie, czy są zmienne do wyświetlenia
    if not categorical_columns_to_include:
        st.write("Brak zmiennych kategorycznych objaśniających z brakami danych.")
    else:
        # **Opis metod eliminacji braków danych:**
        st.markdown("""
        a) Usunięcie wszystkich wierszy z brakującymi wartościami.

        b) Wypełnienie braków w zmiennych kategorycznych najczęściej występującą wartością.

        c) Wypełnienie braków losowymi wartościami z dostępnych w kolumnie.
        """)

        # Wybór kolumny kategorycznej do analizy
        categorical_column_option = st.selectbox("Wybierz kolumnę kategoryczną do analizy:", categorical_columns_to_include)

        # Częstość występowania kategorii dla różnych metod
        freq_original = data[categorical_column_option].value_counts(dropna=False)
        freq_dropped = data_dropped[categorical_column_option].value_counts(dropna=False)
        freq_mode = data_mean[categorical_column_option].value_counts(dropna=False)
        freq_random = data_random[categorical_column_option].value_counts(dropna=False)

        # Łączenie wyników do jednego DataFrame
        freq_combined = pd.DataFrame({
            "Oryginalne (z brakami)": freq_original,
            "Usunięcie wierszy": freq_dropped,
            "Uzupełnienie najczęstszą wartością": freq_mode,
            "Uzupełnienie losowymi wartościami": freq_random
        }).fillna(0)

        # Wizualizacja wyników
        st.write(f"### Porównanie rozkładu kategorii dla kolumny '{categorical_column_option}'")
        fig, ax = plt.subplots(figsize=(10, 6))
        freq_combined.plot(kind="bar", ax=ax)
        ax.set_ylabel("Liczność")
        ax.set_xlabel("Kategoria")
        ax.set_title(f"Porównanie rozkładu kategorii dla kolumny '{categorical_column_option}'")
        st.pyplot(fig)

# Zakładka 3: Modelowanie
with tab3:
    st.write("## Ocena wpływu różnych metod eliminacji braków danych na modelowanie")
    st.write("Wykorzystamy następujące modele klasyfikacyjne:")
    st.markdown("""
    1. **Regresja logistyczna**
    2. **k-Najbliżsi sąsiedzi (k-NN)**
    3. **Naive Bayes**
    """)

    # Sprawdzenie czy zmienna celu jest w danych
    if target_column in data.columns:
        # Zakodowanie zmiennej celu, jeśli jest typu kategorycznego
        y_encoder = LabelEncoder()
        data[target_column] = y_encoder.fit_transform(data[target_column].astype(str))
        class_labels = y_encoder.classes_

        # Przygotowanie danych dla każdej metody
        datasets = {
            'Usunięcie wierszy': data_dropped,
            'Uzupełnienie średnią/trybem': data_mean,
            'Uzupełnienie medianą/trybem': data_median,
            'Uzupełnienie losowymi wartościami': data_random
        }

        # Inicjalizacja wyników
        results = {}

        # Definicja modeli
        models = {
            'Regresja logistyczna': LogisticRegression(max_iter=1000),
            'k-NN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }

        # Iteracja przez metody
        for method_name, dataset in datasets.items():
            # Zakodowanie zmiennej celu w każdym zestawie danych
            dataset[target_column] = y_encoder.transform(dataset[target_column].astype(str))
            y = dataset[target_column]
            X = dataset.drop(columns=[target_column])

            # Remove target_column from categorical_cols
            current_categorical_cols = [col for col in categorical_cols if col != target_column]

            # Kodowanie zmiennych kategorycznych
            X = pd.get_dummies(X, columns=current_categorical_cols, drop_first=True)

            # Upewnienie się, że wszystkie kolumny są numeryczne
            X = X.apply(pd.to_numeric)

            # Skalowanie danych (ważne dla k-NN i regresji logistycznej)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Podział na zbiór treningowy i testowy
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Inicjalizacja wyników dla tej metody
            results[method_name] = {}

            # Iteracja przez modele
            for model_name, model in models.items():
                # Trening modelu
                model.fit(X_train, y_train)

                # Predykcja
                y_pred = model.predict(X_test)

                # Ewaluacja
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                results[method_name][model_name] = report

        # Porównanie wyników w tabelach
        st.write("## Porównanie metryk dla różnych metod eliminacji braków i modeli")

        metrics = ['precision', 'recall', 'f1-score']

        # Iteracja przez klasy
        for class_index, class_label in enumerate(class_labels):
            st.write(f"### Metryki dla klasy '{class_label}'")
            for metric in metrics:
                # Tworzenie tabeli z wynikami
                df_metric = pd.DataFrame(columns=models.keys(), index=datasets.keys())
                for method_name in datasets.keys():
                    for model_name in models.keys():
                        report = results[method_name][model_name]
                        class_key = str(class_index)  # Klucze w raporcie są stringami liczb
                        if class_key in report:
                            metric_value = report[class_key][metric]
                        else:
                            metric_value = np.nan  # Jeśli klasa nie występuje w danych
                        df_metric.loc[method_name, model_name] = metric_value
                # Wyświetlenie tabeli
                st.write(f"#### {metric.capitalize()}")
                st.table(df_metric)        
            st.write(f"Kolumna docelowa '{target_column}' nie została znaleziona w danych.")


st.write("Metoda eliminacji braków danych wpływa na wyniki modeli uczenia maszynowego, choć w przypadku zbioru danych Airlines wpływ ten nie jest drastyczny. Uzupełnianie braków danych pozwala zachować pełny zbiór danych i może prowadzić do lepszej równowagi między precyzją a czułością w modelach.")
st.write("Przeprowadzona analiza Stroke wykazała, że sposób eliminacji braków danych znacząco wpływa na wyniki uzyskiwane po zastosowaniu prostych technik uczenia maszynowego. W przypadku regresji logistycznej usunięcie wierszy zawierających braki danych prowadziło do lepszych wyników, zwłaszcza w zakresie precyzji i F1-score dla obu klas. Uzupełnianie braków danych (średnią, medianą czy losowymi wartościami) często pogarszało wydajność modeli, obniżając kluczowe metryki. Model k-NN wykazywał nieznaczne różnice w wynikach w zależności od metody eliminacji braków, ale ogólnie osiągał lepsze rezultaty po uzupełnieniu braków danych. Naive Bayes prezentował zróżnicowane zachowanie w zależności od zestawu danych, ale również był wrażliwy na wybraną metodę eliminacji braków.")
st.write("Ostateczny wybór metody powinien być uzależniony od specyfiki danych oraz celów analizy. Dlatego ważne jest dobranie odpowiedniej metody eliminacji braków danych w zależności od charakterystyki zbioru danych i celów modelowania.")
