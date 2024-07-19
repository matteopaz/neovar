from lib import write_as_catalog

def classify_periodic(table):
    table["type"] = ["EB"] * len(table)
    return write_as_catalog(table)