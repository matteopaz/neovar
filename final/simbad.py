from astropy import table
from astropy import units as u
import pandas as pd
from astroquery.xmatch import XMatch

VARTYPES = ["EclBin", "LongPeriodV*", "LongPeriodV*_Candidate", "QSO", "RRLyrae", "YSO_Candidate", "Seyfert1", "C*", "YSO", "Mira", "RSCVnV*", "Variable*", "ClassicalCep", "SB*", "Seyfert2", "AGN", "BLLac", "EclBin_Candidate", "AGN_Candidate", "TTauri*", "OrionV*", "BYDraV*", "PulsV*", "Type2Cep", "CataclyV*", "Be*", "delSctV*", "Blazar", "RRLyrae_Candidate", "Seyfert", "RVTauV*", "Cepheid", "Supernova", "Blazar_Candidate", "Eruptive*", "RCrBV*", "Nova", "Supernova_Candidate", "LensedImage", "TTauri*_Candidate", "Planet_Candidate", "IrregularV*", "Ae*", "GravLensSystem", "Variable*_Candidate", "SXPheV*", "alf2CVnV*", "LensedImage_Candidate", "BLLac_Candidate", "Planet", "bCepV*", "Pulsar", "Ae*_Candidate", "Cepheid_Candidate", "LensingEv", "SB*_Candidate", "gammaBurst", "GravLens_Candidate", "RCrBV*_Candidate", "gammaDorV*", "Mira_Candidate", "EllipVar_Candidate"]

def xmatch_simbad(catalog):
    varwise_tbl = table.Table.from_pandas(catalog.reset_index())[["cluster_id", "RAJ2000", "DecJ2000"]]
    simbad = "SIMBAD"
    max_radius = 4
    results = XMatch.query(cat1=varwise_tbl, cat2=simbad, max_distance=max_radius * u.arcsec, colRA1='RAJ2000', colDec1='DecJ2000')[["cluster_id", "main_type"]]
    res = results.to_pandas().set_index("cluster_id")
    res = res.groupby(res.index).first()
    res = res[res["main_type"].isin(VARTYPES)]
    return res
