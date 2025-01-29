import pandas as pd

matches = pd.read_csv("simbad_matches.csv").set_index("cluster_id")

alltypes = matches["main_type"].unique()

VARTYPES = ["EclBin", "LongPeriodV*", "LongPeriodV*_Candidate", "QSO", "RRLyrae", "YSO_Candidate", "Seyfert1", "C*", "YSO", "Mira", "RSCVnV*", "Variable*", "ClassicalCep", "SB*", "Seyfert2", "AGN", "BLLac", "EclBin_Candidate", "AGN_Candidate", "TTauri*", "OrionV*", "BYDraV*", "PulsV*", "Type2Cep", "CataclyV*", "Be*", "delSctV*", "Blazar", "RRLyrae_Candidate", "Seyfert", "RVTauV*", "Cepheid", "Supernova", "Blazar_Candidate", "Eruptive*", "RCrBV*", "Nova", "Supernova_Candidate", "LensedImage", "TTauri*_Candidate", "Planet_Candidate", "IrregularV*", "Ae*", "GravLensSystem", "Variable*_Candidate", "SXPheV*", "alf2CVnV*", "LensedImage_Candidate", "BLLac_Candidate", "Planet", "bCepV*", "Pulsar", "Ae*_Candidate", "Cepheid_Candidate", "LensingEv", "SB*_Candidate", "gammaBurst", "GravLens_Candidate", "RCrBV*_Candidate", "gammaDorV*", "Mira_Candidate", "EllipVar_Candidate"]

for vartype in VARTYPES:
    if vartype not in alltypes:
        print(vartype)