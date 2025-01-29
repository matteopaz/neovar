import json

class_merging = {
    "ea": ["EA"],
    "ew": ["EW"],
    "ecl": ["EB", "ECL"],
    "lpv": ["LPV", "SR", "Mira"],
    "agn": ["agn", "AGN"],
    "rot": ["SOLAR_LIKE", "rotator", "BYDra"],
    "rr": ["RR", "RRab", "RRc"],
    "cep": ["CEP", "DCEP", "T2CEP", "CepII/ACep/CepI", "CEPII", "CepI/CepII", "BCEP", "ACEP", "CepI"],
    "st": ["S", "short-timescale"],
    "rscvn": ["RS", "RSCVN"],
    "dsct": ["DSCT", "DSCT|GDOR|SXPHE"],
    "yso": ["YSO", "BE|GCAS|SDOR|WR"],
    "acv": ["ACV|CP|MCP|ROAM|ROAP|SXARI"],
    "transient": ["CV", "SN"]
}
merger = {} # inverse of class_merging
for key, values in class_merging.items():
    for value in values:
        merger[value] = key

json.dump((merger, list(merger.keys())), open("class_merging.json", "w"))
