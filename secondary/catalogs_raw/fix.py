import pandas as pd

chenneowisetb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/Chen2018-NEOWISE-variables-50282.csv")
chenneowisetb["source_id"] = chenneowisetb["WISE"].astype(str)
chenneowisetb["catalog"] = "chen-neowise"
chenneowisetb = chenneowisetb[["source_id", "catalog", "ra", "dec", "period", "type"]]
chenneowisetb.to_csv("/home/mpaz/neovar/secondary/catalogs/chen-neowise.csv", index=False)

chenztftb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/Chen2020-ZTF-variables-781602.csv")
chenztftb["source_id"] = chenztftb["ZTF"].astype(str)
chenztftb["catalog"] = "chen-ztf"
chenztftb = chenztftb[["source_id", "catalog", "ra", "dec", "period", "type"]]
chenztftb.to_csv("/home/mpaz/neovar/secondary/catalogs/chen-ztf.csv", index=False)

petroskytb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/Petrosky2021-AllWISE-variables-454103.csv")
petroskytb["type"] = petroskytb["type"].map({0: pd.NA, 1: "EB"})
petroskytb.dropna(subset=["type"], inplace=True)
petroskytb["catalog"] = "petrosky"
petroskytb["source_id"] = petroskytb["AllWISE"].astype(str)
petroskytb = petroskytb[["source_id", "catalog", "ra", "dec", "period", "type"]]
petroskytb.to_csv("/home/mpaz/neovar/secondary/catalogs/petrosky.csv", index=False)

gaiacephtb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/GaiaDR3-Cepheids-15021.csv")
gaiacephtb["source_id"] = gaiacephtb["GaiaDR3"].astype(str)
gaiacephtb["catalog"] = "gaia-cepheids"
gaiacephtb = gaiacephtb[["source_id", "catalog", "ra", "dec", "period", "type"]]
gaiacephtb.to_csv("/home/mpaz/neovar/secondary/catalogs/gaia-cepheids.csv", index=False)

gaiaebtb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/GaiaDR3-EclipsingBinaries-2184477.csv")
gaiaebtb["source_id"] = gaiaebtb["GaiaDR3"].astype(str)
gaiaebtb["catalog"] = "gaia-ebs"
gaiaebtb["type"] = "EB"
gaiaebtb = gaiaebtb[["source_id", "catalog", "ra", "dec", "period", "type"]]
gaiaebtb.to_csv("/home/mpaz/neovar/secondary/catalogs/gaia-ebs.csv", index=False)

gaialpvtb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/GaiaDR3-LongPeriodVariables-1720587.csv")
gaialpvtb["source_id"] = gaialpvtb["GaiaDR3"].astype(str)
gaialpvtb["catalog"] = "gaia-lpvs"
gaialpvtb = gaialpvtb[["source_id", "catalog", "ra", "dec", "period", "type"]]
gaialpvtb.to_csv("/home/mpaz/neovar/secondary/catalogs/gaia-lpvs.csv", index=False)

gaiaoscillatortb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/GaiaDR3-MainSequenceOscillators-54476.csv")
gaiaoscillatortb["source_id"] = gaiaoscillatortb["GaiaDR3"].astype(str)
gaiaoscillatortb["catalog"] = "gaia-oscillators"
gaiaoscillatortb["type"] = "oscillator"
gaiaoscillatortb = gaiaoscillatortb[["source_id", "catalog", "ra", "dec", "period", "type"]]
gaiaoscillatortb.to_csv("/home/mpaz/neovar/secondary/catalogs/gaia-oscillators.csv", index=False)

gaiarotatortb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/GaiaDR3-Rotators-474026.csv")
gaiarotatortb["source_id"] = gaiarotatortb["GaiaDR3"].astype(str)
gaiarotatortb["catalog"] = "gaia-rotators"
gaiarotatortb["type"] = "rotator"
gaiarotatortb = gaiarotatortb[["source_id", "catalog", "ra", "dec", "period", "type"]]
gaiarotatortb.to_csv("/home/mpaz/neovar/secondary/catalogs/gaia-rotators.csv", index=False)

gaiarrlyrtb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/GaiaDR3-RRLyraes-271779.csv")
gaiarrlyrtb["source_id"] = gaiarrlyrtb["GaiaDR3"].astype(str)
gaiarrlyrtb["catalog"] = "gaia-rrlyrae"
gaiarrlyrtb = gaiarrlyrtb[["source_id", "catalog", "ra", "dec", "period", "type"]]
gaiarrlyrtb.to_csv("/home/mpaz/neovar/secondary/catalogs/gaia-rrlyrae.csv", index=False)

gaiasptb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/GaiaDR3-ShortTimescales-471679.csv")
gaiasptb["source_id"] = gaiasptb["GaiaDR3"].astype(str)
gaiasptb["catalog"] = "gaia-sts"
gaiasptb["type"] = "short-timescale"
gaiasptb = gaiasptb[["source_id", "catalog", "ra", "dec", "period", "type"]]
gaiasptb.to_csv("/home/mpaz/neovar/secondary/catalogs/gaia-sts.csv", index=False)

spicytb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/spicy.csv")
spicytb["source_id"] = spicytb["SPICY"].astype(str)
spicytb["catalog"] = "spicy"
spicytb["period"] = pd.NA
spicytb["type"] = spicytb["class"]
spicytb = spicytb[["source_id", "catalog", "ra", "dec", "period", "type"]]
spicytb.to_csv("/home/mpaz/neovar/secondary/catalogs/spicy.csv", index=False)

agntb = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/assef-agn.csv")
agntb["source_id"] = agntb["WISE_Designation"].astype(str)
agntb["catalog"] = "assef-agn"
agntb["period"] = pd.NA
agntb["type"] = "agn"
agntb["ra"] = agntb["RA"]
agntb["dec"] = agntb["Dec"]
agntb = agntb[["source_id", "catalog", "ra", "dec", "period", "type"]]
agntb.to_csv("/home/mpaz/neovar/secondary/catalogs/assef-agn.csv", index=False)

gaia_general_catalog = pd.read_csv("/home/mpaz/neovar/secondary/catalogs_raw/Gaia_VariClassifierResult.csv")

conf_cutoff = 0.5
gaia_general_catalog = gaia_general_catalog[gaia_general_catalog["best_class_score"] > conf_cutoff]
print(len(gaia_general_catalog))
gaia_general_catalog["source_id"] = gaia_general_catalog["source_id"].astype(str)
gaia_general_catalog["catalog"] = "gaia-general"
gaia_general_catalog["period"] = pd.NA
gaia_general_catalog["type"] = gaia_general_catalog["best_class_name"]
gaia_general_catalog = gaia_general_catalog[["source_id", "catalog", "ra", "dec", "period", "type"]]
gaia_general_catalog.to_csv("/home/mpaz/neovar/secondary/catalogs/gaia-general.csv", index=False)






