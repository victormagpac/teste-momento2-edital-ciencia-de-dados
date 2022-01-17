import re
import unidecode
from collections import namedtuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def group_separation(row):
    # print(row)
    if row.Significado in ['Sintoma', 'Comorbidade', 'Complicação', 'Histórico pessoal']:
        return row.Significado
    elif row.Coluna in ['nomemacro', 'bairro', 'longitude', 'latitude', 'idh_label', 'sexo', 'idade', 'cor_autorreferida', 'estado_civil', 'escolaridade']:
        return 'Sociodemográfico'
    elif row.Coluna in ['tempo_vmi_total', 'tempo_uti_total', 'tempo_internacao', 'tempo_referencia_internacao', 'tempo_sintomas_internacao']:
        return 'Tempo'
    elif row.Coluna in ['paciente_chegou_com_suporte_respiratorio', 'tipo_caso_a_admissao']:
        return 'Internação'
    elif row.Coluna in ['vmi', 'uti', 'necessidade_transfusional', 'hemodialise']:
        return 'Procedimento'
    elif row.Coluna in ['obito_menos24horas', 'obito_vm', 'obito_uti']:
        return 'Óbito'
    elif row.Coluna == 'desfecho':
        return 'Desfecho'
    else:
        return 'ID'

    
def slugify(text):
    text = unidecode.unidecode(text).upper()
    return re.sub(r'[\W_]+', ' ', text)



def get_columns(df, df_dict, excluded_groups, output='desfecho'):
    excluded_variables = ["tempo_vmi_total", "tempo_uti_total", "tempo_internacao", "longitude", "latitude", "bairro", "paciente_id"]
    reaming_columns = df_dict.query("(Grupo not in @excluded_groups) and (Coluna not in @excluded_variables)").Coluna.to_list()
    bool_columns = df_dict.query("Tipo == 'Booleano' and Coluna in @reaming_columns").Coluna.to_list()
    str_columns = df_dict.query("Tipo == 'Texto' and Coluna in @reaming_columns").Coluna.to_list()
    str_columns.remove(output)
    int_columns = df_dict.query("Tipo == 'Inteiro' and Coluna in @reaming_columns").Coluna.to_list()
    columns = {
        'input': reaming_columns,
        'output': output,
        'boolean': bool_columns,
        'text': str_columns,
        'interger': int_columns
    }
    Columns = namedtuple('Columns', columns)
    return Columns(**columns)

def get_preprocessor(columns):
    boolean_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoding', OneHotEncoder(handle_unknown='ignore', drop='first')),
            ('scaler', StandardScaler(with_mean=False))
        ]
    )

    text_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoding', OneHotEncoder(handle_unknown='ignore')),
            ('scaler', StandardScaler(with_mean=False))
        ]
    )

    interger_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    return ColumnTransformer(
        transformers=[
            ('boolean', boolean_transformer, columns.boolean),
            ('text', text_transformer, columns.text),
            ('interger', interger_transformer, columns.interger)
        ]
    )

def get_Xy(df, columns):
    data = df[columns.input].dropna(subset=[columns.output])
    return (
        data.drop([columns.output], axis=1),
        data[columns.output]
    )