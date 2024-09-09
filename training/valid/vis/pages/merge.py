import os
from PyPDF2 import PdfMerger

# Get a list of all PDF files in the current directory
pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf') and f.split(".")[0] != "merged"]
pdf_files = sorted(pdf_files, key=lambda x: int(x.split(".")[0]))

# Create a PdfFileMerger object
merger = PdfMerger()

# Iterate through the list of PDF files and append each to the PdfFileMerger object
for pdf in pdf_files:
    merger.append(pdf)

# Write the merged PDF to a new file
with open('merged.pdf', 'wb') as output_file:
    merger.write(output_file)

print("All PDF files have been merged into 'merged.pdf'")