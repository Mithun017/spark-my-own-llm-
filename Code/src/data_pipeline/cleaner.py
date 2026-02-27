import os
import re
import glob

def clean_text(text: str) -> str:
    """
    Cleans raw internet text for model training.
    - Removes HTML tags
    - Normalizes white spaces
    - Removes extremely short/useless lines
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove excessive whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Basic unicode normalization (ascii focus for now, or keep clean utf-8)
    # Remove unprintable characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    return text.strip()

def process_raw_data(raw_dir: str, processed_dir: str):
    """
    Reads all .txt files from raw_dir, cleans them, deduplicates paragraphs,
    and saves to processed_dir.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    raw_files = glob.glob(os.path.join(raw_dir, '*.txt'))
    if not raw_files:
        print(f"No raw .txt files found in {raw_dir}")
        return

    print(f"Found {len(raw_files)} raw files. Starting cleaning pipeline...")
    
    unique_paragraphs = set()
    total_cleaned_paragraphs = 0
    
    # We will combine all cleaned unique paragraphs into a single output or maintain file mapping
    # For SPARK SLM, a continuous text format or JSONL works well. Let's do a single master.txt
    master_out_path = os.path.join(processed_dir, "master_dataset.txt")
    
    with open(master_out_path, 'w', encoding='utf-8') as f_out:
        for file_path in raw_files:
            print(f"Processing {os.path.basename(file_path)}...")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                text = f_in.read()
                
                # Split by double newline to get paragraphs
                paragraphs = text.split('\n\n')
                for p in paragraphs:
                    cleaned_p = clean_text(p)
                    # Deduplicate and filter out very short sentences
                    if len(cleaned_p) > 20 and cleaned_p not in unique_paragraphs:
                        unique_paragraphs.add(cleaned_p)
                        f_out.write(cleaned_p + "\n")
                        total_cleaned_paragraphs += 1
                        
    print(f"Done! Cleaned and saved {total_cleaned_paragraphs} unique paragraphs to {master_out_path}")

if __name__ == "__main__":
    # For direct testing
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    process_raw_data(RAW_DIR, PROCESSED_DIR)
