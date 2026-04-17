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

### Public Function: `generate_dataset`

#### Inputs / Outputs

| Type | Variable | Description |
|------|----------|-------------|
| Input | `n` | Number of samples to generate |
| Input | `columns` | Dictionary where keys are column names and values are booleans indicating whether to include them |
| Input | `f` | *Randomness factor*: probability of generating values outside the real-world distribution |
| Input | `progress_callback` | Streamlit progress function used to update dataset generation progress |
| Output | `dataframe` | Generated dataset |

---

### Steps

1. **Load datasets into memory**  
   Only the first sheet of the surname dataset is loaded. [`load_data()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

2. **Preprocess and prepare data**  
   - Clean numeric formats  
   - Build distributions for sampling  
   - Cache results using Streamlit for efficiency  
   [`prepare_all_data()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

3. **Determine generation mode**  
   This defines how **gender distribution** is calculated (most complex dependency).  
   [`get_gender_proportion()`](https://github.com/MAQuesada/DatasetAnonymization/blob/main/src/dataset_creation/creator.py)

4. **Generate each sample**

   4.1. If enabled, update the progress bar periodically  

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

---

## User Interface (UI)

To launch the UI, run the following command from the project root:

```bash
uv run streamlit run src/interface/app_dataset_creation.py --server.port 8502
```

- The application runs on a custom port (8502)
- It provides an interactive form to configure dataset generation

For more details, see the project documentation: [README.md](https://github.com/MAQuesada/DatasetAnonymization/blob/main/README.md)