import pandas as pd


def get_data_train(DATA_PATH=''):
    return pd.read_csv(DATA_PATH+"train.csv")


def get_data_test(DATA_PATH=''):
    return pd.read_csv(DATA_PATH+"test-full.csv")


def clean_predictor(y_pred, Id=None):
    df_test = get_data_test()
    df_train = get_data_train()

    if Id is not None:
        predictions_df = pd.DataFrame(
            {'Id': Id, 'Cover_Type': y_pred})
    else:   # We assume the prediction are sorted
        predictions_df = pd.DataFrame({'Cover_Type': y_pred})
        predictions_df['Id'] = range(1, len(df_test) + 1)

    # Removing those in df_train
    predictions_df.drop(predictions_df[predictions_df["Id"].isin(
        df_train["Id"])].index, inplace=True)

    # Adding df_train instead
    predictions_df = pd.concat(
        [df_train[['Id', 'Cover_Type']], predictions_df], axis=0, ignore_index=True)

    # Sorting by Id (just in case)
    predictions_df.sort_values("Id", inplace=True)

    return predictions_df


def concat_Wilderness_Area(df: pd.DataFrame, new_name="WA_concat", drop_value=False):
    df = df.copy()
    wilderness_areas = [f"Wilderness_Area{i}" for i in range(1, 5)]
    df[new_name] = df[wilderness_areas] @ range(1, 5)
    if drop_value:
        df = df.drop(columns=wilderness_areas)
    return df


def concat_Soil_Type(df: pd.DataFrame, new_name="ST_concat", drop_value=False):
    df = df.copy()
    soil_types = [f"Soil_Type{i}" for i in range(1, 41)]
    df[new_name] = df[soil_types] @ range(1, 41)
    if drop_value:
        df = df.drop(columns=soil_types)
    return df


ST = {
    1: {
        "ELU": 2702,
        "family": "Cathedral",
        "complex": "Rock outcrop complex",
        "stone": "extremely stony",
    },
    2: {
        "ELU": 2703,
        "family": "Vanet",
        "complex": "Ratake families complex",
        "stone": "very stony",
    },
    3: {
        "ELU": 2704,
        "family": "Haploborolis",  # peut etre pas famille..
        "complex": "Rock outcrop complex",
        "stone": "rubbly",
    },
    4: {
        "ELU": 2705,
        "family": "Ratake",
        "complex": "Rock outcrop complex",
        "stone": "rubbly",
    },
    5: {
        "ELU": 2706,
        "family": "Vanet",
        "complex": "Rock outcrop complex",  # complex
        "stone": "rubbly",
    },
    6: {
        "ELU": 2717,
        "family": "Vanet",  # Wetmore families -
        "complex": "Rock outcrop complex",
        "stone": "stony",
    },
    7: {
        "ELU": 3501,
        "family": "Gothic",
        "complex": "",
        "stone": "",
    },
    8: {
        "ELU": 3502,
        "family": "",  # Supervisor -
        "complex": "Limber families complex",
        "stone": "",
    },
    9: {
        "ELU": 4201,
        "family": "Troutville",
        "complex": "",
        "stone": "very stony",
    },
    10: {
        "ELU": 4703,
        "family": "Catamount",  # Bullwark -
        "complex": "Rock outcrop complex",
        "stone": "rubbly",
    },
    11: {
        "ELU": 4704,
        "family": "Catamount",  # Bullwark -
        "complex": "Rock land complex",
        "stone": "rubbly",
    },
    12: {
        "ELU": 4744,
        "family": "Legault",
        "complex": "Rock land complex",
        "stone": "stony",
    },
    13: {
        "ELU": 4758,
        "family": "Catamount",  # - Rock land - Bullwark family complex,
        "complex": "",
        "stone": "rubbly",
    },
    14: {
        "ELU": 5101,
        "family": "",  # Pachic Argiborolis -
        "complex": "Aquolis complex",
        "stone": "",
    },
    15: {
        "ELU": 5151,
        "family": None,
        "complex": None,
        "stone": None,
    },
    16: {
        "ELU": 6101,
        "family": "",  # Cryaquolis - Cryoborolis complex.
        "complex": "",
        "stone": "",
    },
    17: {
        "ELU": 6102,
        "family": "Gateview",  # - Cryaquolis complex.,
        "complex": "",
        "stone": "",
    },
    18: {
        "ELU": 6731,
        "family": "Rogert",
        "complex": "",
        "stone": "very stony",
    },
    19: {
        "ELU": 7101,
        "family": "",  # Typic Cryaquolis - Borohemists complex.,
        "complex": "Typic Cryaquolis",
        "stone": "",
    },
    20: {
        "ELU": 7102,
        "family": "",  # Typic Cryaquepts - Typic Cryaquolls complex.,
        "complex": "Typic Cryaquolis",
        "stone": "",
    },
    21: {
        "ELU": 7103,
        "family": "Leighcan",  # Typic Cryaquolls -, till substratum complex.,
        "complex": "Typic Cryaquolis",
        "stone": "",
    },
    22: {
        "ELU": 7201,
        "family": "Leighcan",  # till substratum,
        "complex": "",
        "stone": "extremely bouldery",
    },
    23: {
        "ELU": 7202,
        "family": "Leighcan",  # till substratum - Typic Cryaquolls complex.,
        "complex": "Typic Cryaquolis",
        "stone": "",
    },
    24: {
        "ELU": 7700,
        "family": "Leighcan",
        "complex": "",
        "stone": "extremely stony",
    },
    25: {
        "ELU": 7701,
        "family": "Leighcan",  # warm,
        "complex": "",
        "stone": "extremely stony",
    },
    26: {
        "ELU": 7702,
        "family": "Catamount",  # Granile -  complex,
        "complex": "",
        "stone": "very stony",
    },
    27: {
        "ELU": 7709,
        "family": "Leighcan",  # warm -
        "complex": "Rock outcrop complex",
        "stone": "extremely stony",
    },
    28: {
        "ELU": 7710,
        "family": "Leighcan",
        "complex": "Rock outcrop complex",
        "stone": "extremely stony",
    },
    29: {
        "ELU": 7745,
        "family": "Como",  # - Legault families complex,
        "complex": "",
        "stone": "extremely stony",
    },
    30: {
        "ELU": 7746,
        "family": "Como",  # - Rock land - Legault family complex,
        "complex": "Rock land complex",
        "stone": "extremely stony",
    },
    31: {
        "ELU": 7755,
        "family": "Catamount",  # Leighcan -  complex,
        "complex": "",
        "stone": "extremely stony",
    },
    32: {
        "ELU": 7756,
        "family": "Catamount",  # - Rock outcrop - Leighcan family complex,
        "complex": "Rock outcrop complex",
        "stone": "extremely stony",
    },
    33: {
        "ELU": 7757,
        "family": "Catamount",  # Leighcan
        "complex": "Rock outcrop complex",
        "stone": "extremely stony",
    },
    34: {
        "ELU": 7790,
        "family": "",  # Cryorthents -
        "complex": "Rock land complex",
        "stone": "extremely stony",
    },
    35: {
        "ELU": 8703,
        "family": "",  # Cryumbrepts - Rock outcrop - Cryaquepts complex.,
        "complex": "Rock outcrop complex",
        "stone": "",
    },
    36: {
        "ELU": 8707,
        "family": "Bross",  # - Rock land - Cryumbrepts complex,
        "complex": "Rock land complex",
        "stone": "extremely stony",
    },
    37: {
        "ELU": 8708,
        "family": "",  # Rock outcrop - Cryumbrepts - Cryorthents complex,
        "complex": "Rock outcrop complex",
        "stone": "extremely stony",
    },
    38: {
        "ELU": 8771,
        "family": "Moran",  # - Leighcan - Cryaquolls complex,
        "complex": "Typic Cryaquolis",
        "stone": "extremely stony",
    },
    39: {
        "ELU": 8772,
        "family": "Moran",  # - Cryorthents - Leighcan family complex,
        "complex": "",
        "stone": "extremely stony",
    },
    40: {
        "ELU": 8776,
        "family": "Moran",  # - Cryorthents -
        "complex": "Rock land complex",
        "stone": "extremely stony",
    },
}


def split_ELU(df: pd.DataFrame, ST=ST):
    df = df.copy()

    climatic_zones = {k: (v['ELU'] // 1000) for k, v in ST.items()}
    df['climatic_zone'] = df['ST_concat'].map(climatic_zones)

    geological_zones = {k: ((v['ELU'] // 100) % 10) for k, v in ST.items()}
    df['geological_zone'] = df['ST_concat'].map(geological_zones)

    last_numbers = {k: v['ELU'] % 100 for k, v in ST.items()}
    df['ELU'] = df['ST_concat'].map(last_numbers)

    family = {k: v['family'] for k, v in ST.items()}
    df['family'] = df['ST_concat'].map(family)

    stone = {k: v['stone'] for k, v in ST.items()}
    df['stone'] = df['ST_concat'].map(stone)

    complex_new = {k: v['complex'] for k, v in ST.items()}
    df['complex'] = df['ST_concat'].map(complex_new)

    return df
