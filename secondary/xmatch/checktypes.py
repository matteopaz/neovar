import pandas as pd
import os 

tmap = {"EA": "ea", "EW": "ew", "EB": "ecl", "ECL": "ecl", "LPV": "lpv", "SR": "lpv", "Mira": "lpv", "agn": "agn", "AGN": "agn", "SOLAR_LIKE": "rot", "rotator": "rot", "BYDra": "rot", "RR": "rr", "RRab": "rr", "RRc": "rr", "CEP": "cep", "DCEP": "cep", "T2CEP": "cep", "CepII/ACep/CepI": "cep", "CEPII": "cep", "CepI/CepII": "cep", "BCEP": "cep", "S": "st", "short-timescale": "st", "RS": "rscvn", "RSCVN": "rscvn", "DSCT": "dsct", "DSCT|GDOR|SXPHE": "dsct", "YSO": "yso", "BE|GCAS|SDOR|WR": "yso", "ACV|CP|MCP|ROAM|ROAP|SXARI": "acv", "CV": "transient", "SN": "transient"}

typecat_pairs = []
for f in list(os.listdir("/home/mpaz/neovar/secondary/catalogs/")):
    catalog = pd.read_csv(f"/home/mpaz/neovar/secondary/catalogs/{f}")
    types = catalog["type"].unique().tolist()
    types = list(set([tmap[t] for t in types if t in tmap.keys()]))
    typecat_pairs.append((f, types))

for pair in typecat_pairs:
    print(pair)