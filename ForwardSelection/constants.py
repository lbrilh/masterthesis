'''
    The numerical and categorical features used in our models. 
    When performing DSL: Include 'source' in categorical_columns
    Else: Exclude
'''

NUMERICAL_COLUMNS = [
    "age",
    "alb",
    "alp",
    "alt",
    "ast",
    "be",
    "bicar",
    "bili",
    "bnd",
    "bun",
    "ca",
    "cai",
    "ck",
    "ckmb",
    "cl",
    "crea",
    "crp",
    "dbp",
    "fgn",
    "fio2",
    "glu",
    "height",
    "hgb",
    "hr",
    "k",
    "lact",
    "lymph",
    "map",
    "mch",
    "mchc",
    "mcv",
    "methb",
    "mg",
    "na",
    "neut",
    "o2sat",
    "pco2",
    "ph",
    "phos",
    "plt",
    "po2",
    "ptt",
    "resp",
    "sbp",
    "temp",
    "tnt",
    "urine",
    "wbc",
    "weight",
]
CATEGORICAL_COLUMNS = ['sex', 'source']
ANCHOR_COLUMNS = ["source"]

LOG_COLUMNS = [
    "alp",
    "alt",
    "ast",
    "bili",
    "bicar",
    "bnd",
    "bun",
    "cai",
    "ck",
    "ckmb",
    "crea",
    "crp",
    "fgn",
    "fio2",
    "glu",
    "hgb",
    "k",
    "lact",
    "lymph",
    "methb",
    "mg",
    "pco2",
    "ph",
    "phos",
    "plt",
    "po2",
    "ptt",
    "tnt",
    "urine",
    "wbc",
]

LINEAR_COLUMNS = [x for x in NUMERICAL_COLUMNS if x not in LOG_COLUMNS]

SOURCE_COLORS = {
    "eicu": "black",
    "mimic": "red",
    "hirid": "blue",
    "miiv": "orange",
    "aumc": "green",
}

KIDNEY_FUNCTION_DYNAMIC_VARIABLES = [
    "alb",
    "alp",
    "alt",
    "ast",
    "be",
    "bicar",
    "bili",
    "bili_dir",
    "bnd",
    "bun",
    "ca",
    "cai",
    "ck",
    "ckmb",
    "cl",
    "crea",
    "crp",
    "dbp",
    "fgn",
    "fio2",
    "glu",
    "hgb",
    "hr",
    "inr_pt",
    "k",
    "lact",
    "lymph",
    "map",
    "mch",
    "mchc",
    "mcv",
    "methb",
    "mg",
    "na",
    "neut",
    "o2sat",
    "pco2",
    "ph",
    "phos",
    "plt",
    "po2",
    "ptt",
    "resp",
    "sbp",
    "temp",
    "tnt",
    "urine",
    "wbc",
]

KIDNEY_FUNCTION_STATIC_VARIABLES = [
    "age",
    "sex",
    "height",
    "weight",
]

LIST_COLUMNS = ["caregiver", "provider", "adm_caregiver", "adm_provider"]


RENAMINGS = {
    "eicu": {
        "patientunitstayid": "stay_id",
        "labresultoffset": "time",
        "hospitalid": "hospital_id",
    },
    "eicu_demo": {
        "patientunitstayid": "stay_id",
        "labresultoffset": "time",
        "hospitalid": "hospital_id",
    },
    "mimic": {
        "icustay_id": "stay_id",
        "hospitalid": "hospital_id",
        "charttime": "time",
    },
    "mimic_demo": {
        "icustay_id": "stay_id",
        "hospitalid": "hospital_id",
        "charttime": "time",
    },
    "hirid": {
        "patientid": "stay_id",
        "datetime": "time",
        "hospitalid": "hospital_id",
    },
    "aumc": {
        "admissionid": "stay_id",
        "measuredat": "time",
    },
    "miiv": {
        "charttime": "time",
    },
}


SOURCES = ["eicu", "mimic", "hirid", "miiv", "aumc"]
