# Dataset Creation - Structure and Functionality

To generate a dataset with a synthetic population of individuals, an application has been developed to create a dataframe based on user-defined parameters. This application is divided into two main parts:

- A **Streamlit UI** where the user can select the desired dataframe characteristics, located at `src/interface/app_dataset_creation.py`
- A **business logic module** responsible for dataset generation, located in `src/dataset_creation`

---

## Business Logic

### Datasets Used

To ensure that the generated data closely resembles real-world distributions, the following datasets from the *Spanish National Statistics Institute (INE)* have been used:

- **[`apellidos_frecuencia.xls`](https://www.ine.es/daco/daco42/nombyapel/apellidos_frecuencia.xls)**  
  Contains an ordered list of the most frequent surnames in Spain as of *January 1, 2024*, with more than 20 occurrences. Each row includes:
  - Surname
  - Number of individuals with it as **first surname**
  - Number of individuals with it as **second surname**
  - Number of individuals with **both surnames identical**

- **[`nombres_por_edad_media_genero.xlsx`](https://www.ine.es/daco/daco42/nombyapel/nombres_por_edad_media.xlsx)**  
  Contains the most common names in Spain (frequency > 20), separated by gender across sheets. Each row includes:
  - Name (simple or compound)
  - Number of individuals with that name
  - *Average age* of individuals with that name

- **[`malagueños_por_generos.csv`](https://www.ine.es/jaxiT3/Tabla.htm?t=2882)**  
  Contains population data for municipalities in Málaga across multiple years. Each row includes:
  - Municipality name and postal code
  - Gender (`Hombre`, `Mujer`, or `Total`)
  - Year of record
  - Total population

- **[`edades_generos_por_paises.csv`](https://www.ine.es/jaxiT3/dlgExport.htm?t=56936)**  
  Contains population distributions by nationality, age group, and gender. Each row includes:
  - Nationality (`Total`, `Española`, etc.)
  - Age group (`De 0 a 4 años`, ..., `Más de 90 años`)
  - Gender (`Total`, `Hombres`, `Mujeres`)
  - Date of record (typically quarterly, from `1 de enero de 2002` to `1 de enero de 2025`)
  - Total population count

All datasets are located at:  
`src/dataset_creation/required_data`

---

### Available Columns for Generation

Based on the dataset characteristics, the following columns can be generated:

| Name | May depend on | Dataset |
|------|--------------|--------|
| **Name** | Gender | `nombres_por_edad_media_genero.xlsx` |
| **First Surname** | — | `apellidos_frecuencia.xls` |
| **Second Surname** | — | `apellidos_frecuencia.xls` |
| **Municipality** | — | `malagueños_por_generos.csv` |
| **Gender** | Municipality, Age *(if Name not selected)*, or independent | `malagueños_por_generos.csv`, `edades_generos_por_paises.csv` |
| **Age** | Name or independent | `nombres_por_edad_media_genero.xlsx`, `edades_generos_por_paises.csv` |

---

### Custom Columns for Generation

In addition to the default columns, the user can create **as many custom columns as desired**.

There are two supported types of custom columns:

#### Categoricals

| Required fields | Description | 
|-----------------|-------------|
| **Column name** | Name of the generated column |
| **List of values** | Comma-separated list of possible category values |
| **List of weights** | Comma-separated list of weights corresponding to each value |

**Notes:**
- The number of weights should match the number of values *(or be one less)*  
- The sum of weights should be approximately **1**  
- If the weights are invalid or missing, a **uniform distribution** will be applied 

#### Numerical Columns

| Required fields | Description |
|-----------------|-------------|
| **Column name** | Name of the generated column |
| **Low value** | Minimum value that can be generated | 
| **High value** | Maximum value that can be generated | 
| **Distribution** | Statistical distribution used to generate the values |

**Supported distributions:**
- `uniform`
- `normal`
- `binomial`
- `lognormal`

---

### Public Function: `generate_dataset`

#### Inputs / Outputs

| Type | Variable | Description |
|------|----------|-------------|
| Input | `n` | Number of samples to generate |
| Input | `columns` | Dictionary where keys are column names and values are booleans indicating whether to include them |
| Input | `f` | *Randomness factor*: probability of generating values outside the real-world distribution |
| Input | `custom_columns` | List of dictionaries defining custom columns |
| Input | `progress_callback` | Function used to update progress (e.g., Streamlit progress bar) |
| Output | `dataframe` | Generated dataset |

**Custom Columns Format**

```python
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
```

---

### Steps

#### Step 1: Individual Data Generation (`generate_dataset_indv()`)

1. **Load datasets into memory**  
   - Load all required datasets into memory
   - Only the first sheet of the surname dataset is loaded. 
   [`load_data()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

2. **Preprocess and prepare data**  
   - Clean numeric formats  
   - Build distributions for sampling  
   - Cache results using Streamlit for efficiency  
   [`prepare_all_data()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

3. **Determine generation mode**  
   This defines how **gender distribution** is calculated (most complex dependency):
    - Based on municipality
    - Based on age (if name is not selected)
    - Or global distribution
   [`get_gender_proportion()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

4. **Generate each sample**

   4.1. **Progress update (optional)**. If enabled, update the progress bar periodically  

   4.2. **Municipality generation**  
   Generated using weighted sampling from real distribution  
   [`generate_municipality()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

   4.3. **Age generation (independent case: no name)**  
   Based on Spanish population distribution.   
   [`generate_age_general()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

   4.4. **Gender generation**  
   Depends on:
   - Municipality  
   - Age  
   - Or global distribution  
   [`generate_gender()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

    *Note*: Gender may be generated even if not included, as it is required for name generation.

   4.5. **Name generation**  
   - Based on gender distribution  
   - If age is required → generated from name mean age using Gaussian distribution  
   [`generate_name()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py), [`generate_age_from_name()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

   4.6. **First surname generation**  
   [`generate_surname()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

   4.7. **Second surname generation**  
   [`generate_surname()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

   4.8. Store generated values as a JSON-like structure in a list

5. Convert the list into a dataframe and return it

#### Step 2: Custom Columns Generation

1. **Iterate over custom columns.** For each column:

   1.1. **Categorical** --> A column dataframe is generated using weighted sampling. (`generate_random_categories()`)

   1.2. **Numerical** --> A column dataframe is generated using the specified distribution. (`generate_random_numbers()`)

2. **Merge results.** Each generated column is concatenated to the main dataframe. 

---

## User Interface (UI)

To launch the UI, run the following command from the project root:

```bash
uv run streamlit run src/interface/ui_creation/app_dataset_creation.py --server.port 8502
```

- The application runs on a custom port (8502)
- It provides an interactive form to configure dataset generation

For more details, see the project documentation: [README.md](https://github.com/MAQuesada/DatasetAnonymization/blob/main/README.md)