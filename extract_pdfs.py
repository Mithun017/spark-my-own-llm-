import os
import sys

# Ensure pypdf is installed
try:
    import pypdf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    import pypdf

doc_dir = r"c:\Users\MITHUN\Desktop\STUDIES\PROJECT\42.SPARK - My own slm\Doc"
pdfs = [
    "0.The_Brain_of_the_spark.pdf",
    "1.spark_slm_general.pdf",
    "2.spark_slm_architecture_and_components.pdf",
    "3.spark_slm_training_and_data_pipeline.pdf",
    "4.spark_slm_inference_optimization_scaling.pdf",
    "5.spark_slm_advanced_extensions_safety_blueprint.pdf",
    "6.All in here complety.pdf"
]

all_text = ""
for pdf in pdfs:
    path = os.path.join(doc_dir, pdf)
    print(f"Reading {pdf}...")
    try:
        reader = pypdf.PdfReader(path)
        text = f"\n\n================ {pdf} ================\n\n"
        for i, page in enumerate(reader.pages):
            t = page.extract_text()
            if t:
                text += t + "\n"
        print(f"Read {pdf}, extracted {len(text)} characters.")
        all_text += text
    except Exception as e:
        print(f"Error reading {pdf}: {e}")

output_path = r"c:\Users\MITHUN\Desktop\STUDIES\PROJECT\42.SPARK - My own slm\extracted_doc.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(all_text)
print(f"Finished writing extracted text to {output_path}")
