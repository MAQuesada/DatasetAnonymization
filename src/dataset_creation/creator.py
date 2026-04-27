import numpy as np
import pandas as pd
import random
from pathlib import Path
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
LAST_YEAR = 2024  # Last registred year


# =========================== LOAD DATA =========================
def load_data():
    """
    Load all required dataset to generate the data. It returns 5 datasets in the format dataframe

    Output: commun male names, commun female names, commun surnames, municipality from malaga, ages in Spain
    """
    names_male = pd.read_excel(
        BASE_DIR / "required_data/nombres_por_edad_media_genero.xlsx",
        sheet_name=0,
        skiprows=6,
    )

    names_female = pd.read_excel(
        BASE_DIR / "required_data/nombres_por_edad_media_genero.xlsx",
        sheet_name=1,
        skiprows=6,
    )

    surnames = pd.read_excel(
        BASE_DIR / "required_data/apellidos_frecuencia.xls", skiprows=4, sheet_name=0
    )

    malaga = pd.read_csv(BASE_DIR / "required_data/malagueños_por_generos.csv", sep=";")
    ages = pd.read_csv(
        BASE_DIR / "required_data/edades_generos_por_paises.csv", sep=";"
    )

    return names_male, names_female, surnames, malaga, ages


# ======================= PREPARE DATA TO USE IT AT THE MOMENT ===================================
# Clean numeric columns (remove dots as thousand separators)
def clean_numeric(col):
    """
    Given a numeric column in string format, remove dots as thousand separators and transform the value to float
    """
    return (
        col.astype(str).str.replace(".", "", regex=False).replace("", "0").astype(float)
    )


# Clean numeric columns (change commas to dots as decimal separators)
def clean_comma(col):
    """
    Given a numeric column in string format, replace commas as decimal separators to dots and transform the value to float
    """
    return (
        col.astype(str)
        .str.replace(",", ".", regex=False)
        .replace("", "0")
        .astype(float)
    )


def prepare_names(names_male, names_female):
    """
    Converts the names dataframes in a distribution sample and store them in a dictionary,
    where males are with the "Hombre" key and females with "Mujer" key. These keys are in
    Spanish to facilitate the selection depending on Gender value.

    Output: Dictionary as

    {
        Hombre: [('ANTONIO', 603004.0), ('MANUEL', 534398.0), ('JOSE', 501731.0), ...],

        Mujer: [('MARIA CARMEN', 624368.0), ('MARIA', 551105.0), ('CARMEN', 340920.0), ...]
    }
    """

    # Remove dots and separators
    names_male["Frecuencia"] = clean_numeric(names_male["Frecuencia"])
    names_female["Frecuencia"] = clean_numeric(names_female["Frecuencia"])

    #  Hombre (man) and Mujer (woman) are in Spanish because it would employed to find the frequency
    return {
        "Hombre": (names_male["Nombre"].tolist(), names_male["Frecuencia"].tolist()),
        "Mujer": (names_female["Nombre"].tolist(), names_female["Frecuencia"].tolist()),
    }


def prepare_surnames(surnames_df):
    """
    Converts surname dataframe in two distribution samples, split up in first surname and second surname.

    Output: surname1 sample, surname2 sample.

    Example:
    - Surname1 distribution sample: [('GARCIA', 1449151.0), ('RODRIGUEZ', 935440.0), ...]
    - Surname2 distribution sample: [('GARCIA', 1469986.0), ('RODRIGUEZ', 945899.0), ...]
    """
    # Surnames would appear as Total.1, etc. To facilitate the processed data, we rename the columns
    surnames_df.columns = [
        "orden",
        "apellido",
        "freq_apellido1",
        "freq_apellido2",
        "freq_ambos",
        "freq_ambos_repetida",  # Due to a cell combination, it appears double
    ]

    surnames_df["freq_apellido1"] = clean_numeric(surnames_df["freq_apellido1"])
    surnames_df["freq_apellido2"] = clean_numeric(surnames_df["freq_apellido2"])

    # Transform 9999999 to frequency 0
    surnames_df = surnames_df.replace(9999999, 0)

    # Extract distributions
    surnames = surnames_df["apellido"].tolist()
    freq1 = surnames_df["freq_apellido1"].tolist()
    freq2 = surnames_df["freq_apellido2"].tolist()

    return (surnames, freq1), (surnames, freq2)


def prepare_age_by_name(names_male, names_female):
    """
    Prepare the age by name distribution sample, merging the gender dataset and change the comma separator to dot

    Output, example:

    [('ANTONIO', 58.3), ('MANUEL', 56.3), ('JOSE', 44.5), ...]
    """
    df = pd.concat([names_male, names_female])

    df["Edad Media (*)"] = clean_comma(df["Edad Media (*)"])

    return dict(zip(df["Nombre"], df["Edad Media (*)"]))


def prepare_municipalities(malaga_df):
    """
    Prepares the municipalities dataset to be employed to find Gender, filtering the data by the last year we have data (2024, given by names dataset).
    Also prepares a distribution example about municipalities from Malaga to be employed.

    Output: processed municipalities dataframe and distribution sample. Examples:

    Index,  Municipios,  Sexo,  Periodo,   Total

    1,   29001 Alameda,    Total,     2024,  5449.0

    31,  29001 Alameda,  Hombres,     2024,  2750.0

    61,  29001 Alameda,  Mujeres,     2024,  2699.0

    Municipality distribution sample: [('29001 Alameda', 5449.0), ('29002 Alcaucín', 2653.0), ('29003 Alfarnate', 1046.0), ...]
    """

    # We only need the 2024 data, because it is the most recent respect to name and surname list (1/1/2024)
    df = malaga_df[malaga_df["Periodo"] == LAST_YEAR].copy()

    df["Total"] = clean_numeric(df["Total"])

    # Also we create the distribution sample
    municipality_df = df[df["Sexo"] == "Total"]
    municipality_dist = (
        municipality_df["Municipios"].tolist(),
        municipality_df["Total"].tolist(),
    )

    return df, municipality_dist


def prepare_age_default(ages_df):
    """
    Process the dataframe with nationality==Spanish and period==1 January 2024. Also transform the numeric columns to the correct format.

    Output: Dataframe as:

    Index,Nacionalidad, Grupo quinquenal de edad,     Sexo,             Periodo,       Total

    5584,     Española,         Todas las edades,    Total,  1 de enero de 2024,  42117413.0

    5677,     Española,         Todas las edades,  Hombres,  1 de enero de 2024,  20572893.0

    5770,     Española,         Todas las edades,  Mujeres,  1 de enero de 2024,  21544520.0

    """
    # Filter by nationality and the date
    age_preprocessed = ages_df[
        (ages_df["Nacionalidad"] == "Española")
        & (ages_df["Periodo"] == "1 de enero de " + str(LAST_YEAR))
    ].copy()

    age_preprocessed["Total"] = clean_numeric(age_preprocessed["Total"])

    return age_preprocessed


# ================== SPECIAL CASES OF PROPORTION ===========
# ------------------GENDER----------------------------
def get_gender_proportion(malaga_df, ages_df, mode):
    """
    Decide gender distribution source based on user-selected columns:

    - If municipality (mode == municipality) → Gender depends on municipality
    - if not name and age (mode == age) → Gender depends on age
    - otherwise → Gender depends on nothing

    """

    if mode == "municipality":
        municipalities = malaga_df["Municipios"].unique()

        result = {}
        for m in municipalities:
            sub = malaga_df[malaga_df["Municipios"] == m]
            men = sub[sub["Sexo"] == "Hombres"]["Total"].sum()
            women = sub[sub["Sexo"] == "Mujeres"]["Total"].sum()
            total = men + women
            if total > 0:
                result[m] = (men / total, women / total)

    elif mode == "age":
        df = ages_df[(ages_df["Sexo"].isin(["Hombres", "Mujeres"]))]

        result = {}
        for group in df["Grupo quinquenal de edad"].unique():
            sub = df[df["Grupo quinquenal de edad"] == group]
            men = sub[sub["Sexo"] == "Hombres"]["Total"].sum()
            women = sub[sub["Sexo"] == "Mujeres"]["Total"].sum()
            total = men + women
            if total > 0:
                result[group] = (men / total, women / total)

    else:
        df = ages_df[(ages_df["Grupo quinquenal de edad"] == "Todas las edades")]

        men = df[df["Sexo"] == "Hombres"]["Total"].sum()
        women = df[df["Sexo"] == "Mujeres"]["Total"].sum()
        total = men + women

        result = {"Total": (men / total, women / total)}

    return result


# -------------- AGE WHEN WE DONT USE MUNICIPALITY AND NAME ----------------
def get_age_proportion(ages_df):
    """
    Processed the age dataframe to be independet to name and gender. Transform it to a distribution sample.

    Output example:

    [('De 0 a 5 años', 1487998), ...]
    """
    df = ages_df[
        (ages_df["Sexo"] == "Total")
        & (ages_df["Grupo quinquenal de edad"] != "Todas las edades")
    ]

    return (df["Grupo quinquenal de edad"].tolist(), df["Total"].tolist())


# ========== SELECT RANDOM VALUE OR A NORMAL PROPORTION VALUE ========
def mixed_sample(values, weights, f):
    if random.random() < f:
        return random.choice(values)
    else:
        return random.choices(values, weights=weights, k=1)[0]


# ========================== GENERATORS ==========================
def generate_gender(p_men, p_women, f):
    return mixed_sample(["Hombre", "Mujer"], [p_men, p_women], f)


def generate_name(name_dist, gender, f):
    values, weights = name_dist[gender]
    return mixed_sample(values, weights, f)


def generate_surname(surname_dist, f):
    values, weights = surname_dist
    return mixed_sample(values, weights, f)


def generate_municipality(municipality_dist, f):
    values, weights = municipality_dist
    return mixed_sample(values, weights, f)


# -- AGE GENERATORS --> IT DEPENDS ON NAME OR ANYTHING ------
def compute_sigma(mean):
    if mean < 20:
        return 5
    elif mean < 40:
        return 10
    elif mean < 60:
        return 12
    else:
        return 15


def generate_age_from_name(name, age_dict, f):
    mean = age_dict.get(name, 40)

    if random.random() < f:
        return random.randint(0, 100)
    else:
        sigma = compute_sigma(mean)
        age = int(random.gauss(mean, sigma))
        return max(0, min(100, age))


def generate_age_general(ages_dist, f):
    values, weight = ages_dist
    group = mixed_sample(values, weight, f)

    # Convert text range to number
    if "y más" in group:
        age = random.randint(90, 100), group
    else:
        low, high = map(int, group.replace("De ", "").replace(" años", "").split(" a "))
        age = random.randint(low, high), group

    return age


# =================== PREPARE DATA =========================================


# This data is always static in each dataset generation. To avoid overflow the page,
# we are going to say that it stays in cache all the time
@st.cache_data
def prepare_all_data():
    """
    Load the required dataframe to generate dataset and also prepare all the independent distribution
    and dataframes respect the user actions. In this way, we haven't to redo the operation each time.

    Output:
     - Name distribution sample
     - First and Second surnames distribution sample
     - Map between name and mean age
     - Malaga dataframe and municipality distribution
     - Age dataframe
     - Age distribution without dependencies
    """
    # 1. Load dataset
    names_male, names_female, surnames_df, malaga_df, ages_df = load_data()

    # 2. Prepare dataset and dictionaries
    name_dist = prepare_names(names_male, names_female)
    age_name_map = prepare_age_by_name(names_male, names_female)
    surname1_dist, surname2_dist = prepare_surnames(surnames_df)
    malaga_df, municipality_dist = prepare_municipalities(malaga_df)
    ages_df = prepare_age_default(ages_df)
    age_dist = get_age_proportion(ages_df)

    return (
        name_dist,
        age_name_map,
        surname1_dist,
        surname2_dist,
        malaga_df,
        municipality_dist,
        ages_df,
        age_dist,
    )


# =================== MAIN FUNCTION --> CREATE INDIVIDUALS ============================================
def generate_dataset_indv(n, columns, f, progress_callback=None):
    """
    INPUTS:
        - n => Number of samples we want to generate on dataset
        - columns => Kind of columns we want to generate
        - f => Random distribution of samples
    OUTPUTS:
        - Generated dataframe

    A dataframe is generated depending on some real data from INE. The user can select
    the following column:

        - Name. If it is selected, Age would be depend on it. Name depends on Gender. Use "nombres_por_edad_media.xlsx" file
        - 1º surname. Use "apellidos_frecuencia.xls" to be generated.
        - 2º surname. Use "apellidos_frecuencia.xls" to be generated.
        - Gender. Depends on municipalities and, if the Municipality and Name aren't selected, it will depend on Age. Use "edades_generos_por_paises.csv" or "malagueños_por_generos.csv" files
        - Age. Depends on Name if it is selected. Otherwise, it depends on nothing. Use "nombres_por_edad_media.xlsx" or "edades_generos_por_paises.csv" files
        - Municipality. Use "malagueños_por_generos.csv" to be generated.

    """
    # 1. Load dataset and prepare it along the distribution samples. They are going to be cached in memory
    (
        name_dist,
        age_name_map,
        surname1_dist,
        surname2_dist,
        malaga_df,
        municipality_dist,
        ages_df,
        age_dist,
    ) = prepare_all_data()

    # 2. To avoid continouos access to memory, we store it in a variable
    use_name = columns.get("name", False)
    use_surname1 = columns.get("surname1", False)
    use_surname2 = columns.get("surname2", False)
    use_gender = columns.get("gender", False)
    use_age = columns.get("age", False)
    use_municipality = columns.get("municipality", False)

    # 3. Get age and gender proportion
    # Determine dependency mode
    if use_municipality:
        mode = "municipality"
    elif not use_name and use_age:
        mode = "age"
    else:
        mode = "Total"
    gender_info = get_gender_proportion(malaga_df, ages_df, mode)

    data = []

    # 4. We start to generate data on each row
    for i in range(n):
        if progress_callback and i % 10 == 0:
            progress_callback(i / n)

        row = {}

        # Aux Variables to know gender dependencies
        municipality = None
        gender = None

        # GENERATE Municipality
        if use_municipality:
            municipality = generate_municipality(municipality_dist, f)
            row["Municipality"] = municipality

        # If mode == age --> GENERATE Age (We need it before gender)
        age = None
        age_group = None
        if mode == "age":
            age, age_group = generate_age_general(age_dist, f)
            row["Age"] = age

        # GENERATE Gender according to dependencies
        if use_name or use_gender:
            if mode == "municipality":  # Municipality --> Gender
                p_m, p_w = gender_info.get(municipality, (0.5, 0.5))
            elif mode == "age":  # Age --> Gender
                p_m, p_w = gender_info.get(age_group, (0.5, 0.5))
            else:  # Gender --> Name --> Age  or  Gender because Age isn't required.
                p_m, p_w = gender_info["Total"]

            gender = generate_gender(p_m, p_w, f)

            if use_gender:
                row["Gender"] = gender

        # GENERATE Name
        if use_name:
            name = generate_name(name_dist, gender, f)
            row["Name"] = name

            # GENERATE Age
            if use_age:
                row["Age"] = generate_age_from_name(name, age_name_map, f)

        # If no name and not mode == "age" ==> GENERATE Age
        if use_age and "Age" not in row:
            age, _ = generate_age_general(age_dist, f)
            row["Age"] = age

        # GENERATE surname1
        if use_surname1:
            surname1 = generate_surname(surname1_dist, f)
            row["Surname1"] = surname1

        # GENERATE surname2
        if use_surname2:
            surname2 = generate_surname(surname2_dist, f)
            row["Surname2"] = surname2

        data.append(row)

    return pd.DataFrame(data)


# =================== MAIN FUNCTION --> CREATE RANDOM COLUMNS ============================================


# -----------------CATEGORIES--------------------------
def generate_random_categories(n, name_col, pos_val, f, weights=None):
    """
    INPUTS:
        - n => Number of samples we want to generate on dataset
        - name_col => Column name we want to generate
        - pos_val => Possible categories
        - f => Random distribution of samples
        - weights => Appearencies proportion of the possible categories
    OUTPUTS:
        - Generated dataframe with only a column

    Given the previous values, it generates a dataframe where each row is generated randomly using the weights proportion
    and the specified possible values. If the proportion is wrong, an equal proportion will be used.
    """
    sz_pv = len(pos_val)
    if weights is None or len(weights) < sz_pv - 1 or abs(sum(weights) - 1) > 1e-6:
        # If there isnt any proportion or some values are missing, we will use an equal proportion
        weights = [1 / sz_pv for _ in range(sz_pv)]
    elif len(weights) == sz_pv - 1:
        weights.append(1 - sum(weights))
    elif len(weights) > sz_pv:
        weights = weights[:sz_pv]

    data = [mixed_sample(pos_val, weights, f) for _ in range(n)]

    return pd.DataFrame({name_col: data})


# ------------ NUMERICAL ----------------------
def generate_random_numbers(n, name_col, low_v, sup_v, f, distribution="normal"):
    """
    INPUTS:
        - n => Number of samples we want to generate on dataset
        - name_col => Column name we want to generate
        - low_v => Minimum possible value to be generated
        - sup_v => Maximum possible value to be generated
        - f => Random distribution of samples
        - distribution (["normal" | "uniform" | "binomial", | "lognormal"]) => String indicating the kind of numeric distribution will be aplied
    OUTPUTS:
        - Generated dataframe with only a column

    Given the previous values, it generates a dataframe where each row is generated randomly using the specified distribution.
    If it is not specified, a normal distribution will be used.
    """
    if distribution == "uniform":
        data = np.random.uniform(low_v, sup_v, n)

    elif distribution == "normal":
        mean = (low_v + sup_v) / 2
        std = (sup_v - low_v) / 6
        data = np.random.normal(mean, std, n)

    elif distribution == "binomial":
        p = 0.5
        trials = int(sup_v)
        data = np.random.binomial(trials, p, n)

    elif distribution == "lognormal":
        mean = (low_v + sup_v) / 2
        sigma = 1
        data = np.random.lognormal(mean, sigma, n)

    else:
        raise ValueError(f"Distribution '{distribution}' no supported")

    # Obtains random values and random index creating a n list random from 0 to 1
    noise_mask = np.random.rand(n) < f
    noise_values = np.random.uniform(low_v, sup_v, n)

    # Replace
    data[noise_mask] = noise_values[noise_mask]

    # Ensure the range
    data = np.clip(data, low_v, sup_v)

    return pd.DataFrame({name_col: data})


def generate_dataset(n, columns, f, custom_columns=None, progress_callback=None):
    """
    INPUTS:
        - n => Number of samples we want to generate on dataset
        - columns => Kind of columns we want to generate
        - f => Random distribution of samples
    OUTPUTS:
        - Generated dataframe

    A dataframe is generated depending on some real data from INE. The user can select
    the following column:

        - Name. If it is selected, Age would be depend on it. Name depends on Gender. Use "nombres_por_edad_media.xlsx" file
        - 1º surname. Use "apellidos_frecuencia.xls" to be generated.
        - 2º surname. Use "apellidos_frecuencia.xls" to be generated.
        - Gender. Depends on municipalities and, if the Municipality and Name aren't selected, it will depend on Age. Use "edades_generos_por_paises.csv" or "malagueños_por_generos.csv" files
        - Age. Depends on Name if it is selected. Otherwise, it depends on nothing. Use "nombres_por_edad_media.xlsx" or "edades_generos_por_paises.csv" files
        - Municipality. Use "malagueños_por_generos.csv" to be generated.

    Moreover, the user can add some custom columns. These ones have the following format:
    [
        {
            "type": "categorical",
            "name": "Job",
            "values": ["Engineer", "Doctor", "Teacher"],
            "weights": [0.5, 0.3, 0.2],
        },
        {
            "type": "numerical",
            "name": "Salary",
            "low": 1000,
            "high": 5000,
            "distribution": "normal",
        }
    ]
    """

    df = generate_dataset_indv(n, columns, f, progress_callback)
    if custom_columns:
        for col in custom_columns:
            if col["type"] == "categorical":
                new_df = generate_random_categories(
                    n,
                    col["name"],
                    col["values"],
                    f,
                    col.get("weights"),
                )

            elif col["type"] == "numerical":
                new_df = generate_random_numbers(
                    n,
                    col["name"],
                    col["low"],
                    col["high"],
                    f,
                    col.get("distribution", "normal"),
                )

            df = pd.concat([df, new_df], axis=1)
    
    return df 

