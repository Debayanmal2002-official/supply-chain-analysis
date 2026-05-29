import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
file_path = BASE_DIR / "data" / "APL_Logistics.parquet"
df = pd.read_parquet(file_path)

## Dropping Unnecessary
drop_cols = [
    'Customer Fname',
    'Customer Lname',
    'Customer Street',
    'Customer Zipcode',
    'Category Id',
    'Department Id',
    'Order Customer Id'
]
df = df.drop(columns=drop_cols)
df.rename(columns={'Type': 'Payment Type'}, inplace=True)

## Fixing Messy Dataset
obj_cols = df.select_dtypes(include='object').columns
for col in obj_cols:
    df[col] = df[col].str.strip()
df = df[~df['Customer State'].isin(['91732', '95758'])]
df['Order Region'] = df['Order Region'].replace({'South of  USA ':'South of USA'})
country_map = {
    'Afganistán': 'Afghanistan',
    'Alemania': 'Germany',
    'Arabia Saudí': 'Saudi Arabia',
    'Argelia': 'Algeria',
    'Azerbaiyán': 'Azerbaijan',
    'Bangladés': 'Bangladesh',
    'Baréin': 'Bahrain',
    'Belice': 'Belize',
    'Bélgica': 'Belgium',
    'Benín': 'Benin',
    'Bielorrusia': 'Belarus',
    'Bosnia y Herzegovina': 'Bosnia and Herzegovina',
    'Botsuana': 'Botswana',
    'Brasil': 'Brazil',
    'Burkina Faso': 'Burkina Faso',
    'Bután': 'Bhutan',
    'Camboya': 'Cambodia',
    'Camerún': 'Cameroon',
    'Chipre': 'Cyprus',
    'Corea del Sur': 'South Korea',
    'Costa de Marfil': 'Ivory Coast',
    'Croacia': 'Croatia',
    'Dinamarca': 'Denmark',
    'EE. UU.': 'United States',
    'Egipto': 'Egypt',
    'Emiratos Árabes Unidos': 'United Arab Emirates',
    'Eslovaquia': 'Slovakia',
    'Eslovenia': 'Slovenia',
    'España': 'Spain',
    'Estados Unidos': 'United States',
    'Etiopía': 'Ethiopia',
    'Filipinas': 'Philippines',
    'Finlandia': 'Finland',
    'Francia': 'France',
    'Gabón': 'Gabon',
    'Grecia': 'Greece',
    'Guadalupe': 'Guadeloupe',
    'Guayana Francesa': 'French Guiana',
    'Guinea Ecuatorial': 'Equatorial Guinea',
    'Guinea-Bissau': 'Guinea-Bissau',
    'Hungría': 'Hungary',
    'Irak': 'Iraq',
    'Irán': 'Iran',
    'Irlanda': 'Ireland',
    'Italia': 'Italy',
    'Japón': 'Japan',
    'Jordania': 'Jordan',
    'Kazajistán': 'Kazakhstan',
    'Kenia': 'Kenya',
    'Kirguistán': 'Kyrgyzstan',
    'Laos': 'Laos',
    'Lesoto': 'Lesotho',
    'Líbano': 'Lebanon',
    'Lituania': 'Lithuania',
    'Luxemburgo': 'Luxembourg',
    'Malasia': 'Malaysia',
    'Marruecos': 'Morocco',
    'México': 'Mexico',
    'Moldavia': 'Moldova',
    'Mozambique': 'Mozambique',
    'Myanmar (Birmania)': 'Myanmar',
    'Níger': 'Niger',
    'Noruega': 'Norway',
    'Nueva Zelanda': 'New Zealand',
    'Omán': 'Oman',
    'Pakistán': 'Pakistan',
    'Panamá': 'Panama',
    'Papúa Nueva Guinea': 'Papua New Guinea',
    'Países Bajos': 'Netherlands',
    'Perú': 'Peru',
    'Polonia': 'Poland',
    'Reino Unido': 'United Kingdom',
    'República Centroafricana': 'Central African Republic',
    'República Checa': 'Czech Republic',
    'República de Gambia': 'Gambia',
    'República del Congo': 'Republic of the Congo',
    'República Democrática del Congo': 'Democratic Republic of the Congo',
    'República Dominicana': 'Dominican Republic',
    'Ruanda': 'Rwanda',
    'Rumania': 'Romania',
    'Rusia': 'Russia',
    'Sáhara Occidental': 'Western Sahara',
    'Sierra Leona': 'Sierra Leone',
    'Singapur': 'Singapore',
    'Sudán': 'Sudan',
    'Sudán del Sur': 'South Sudan',
    'Suecia': 'Sweden',
    'Suazilandia': 'Eswatini',
    'Suiza': 'Switzerland',
    'Surinam': 'Suriname',
    'Taiwán': 'Taiwan',
    'Tailandia': 'Thailand',
    'Tanzania': 'Tanzania',
    'Tayikistán': 'Tajikistan',
    'Trinidad y Tobago': 'Trinidad and Tobago',
    'Turkmenistán': 'Turkmenistan',
    'Turquía': 'Turkey',
    'Túnez': 'Tunisia',
    'Ucrania': 'Ukraine',
    'Uzbekistán': 'Uzbekistan',
    'Yibuti': 'Djibouti',
    'Zimbabue': 'Zimbabwe'
}
df['Order Country'] = df['Order Country'].replace(country_map)
df['Order Status'] = df['Order Status'].str.replace('_', ' ').str.title()
df['Customer Country'] = df['Customer Country'].replace({
    'EE. UU.': 'United States'
})

for col in obj_cols:
    df[col] = df[col].str.strip()
    print(f"Column: {col}")
    print(f"\nTotal Unique Values: {df[col].nunique()}")
    print("\n")

df = df.drop_duplicates()

csv_path = BASE_DIR / "data" / "APL_Logistics_Clean.csv"
parquet_path = BASE_DIR / "data" / "APL_Logistics_Clean.parquet"
df.to_csv(csv_path, index=False)
df.to_parquet(parquet_path, index=False)
