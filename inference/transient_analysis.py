from lib import write_as_catalog

def classify_transient(table):
    table["type"] = ["Nova"] * len(table)
    return write_as_catalog(table)