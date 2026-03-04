import os
import requests
from bs4 import BeautifulSoup
import re

def scrape_article(url, output_path):
    print(f"Scraping: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from paragraphs and pre/code blocks
        content_elements = soup.find_all(['p', 'pre', 'code'])
        
        extracted_text = []
        for el in content_elements:
            text = el.get_text(separator='\n').strip()
            # Basic cleanup
            text = re.sub(r'\n+', '\n', text)
            if len(text) > 30: # Ignore tiny useless snippets
                extracted_text.append(text)
                
        if extracted_text:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(extracted_text))
            print(f"Successfully saved {len(extracted_text)} blocks to {output_path}")
        else:
            print("No significant content found.")
            
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

def run_spider(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    
    # 1. Clean out the old Shakespeare educational dataset
    old_data = os.path.join(raw_dir, "tiny_shakespeare.txt")
    if os.path.exists(old_data):
        print("Removing old tiny_shakespeare dataset to focus on new intelligent data...")
        os.remove(old_data)
        
    # 2. Targeted urls for crawling (A mix of general intelligence and coding)
    TARGETS = [
        ("https://en.wikipedia.org/wiki/Artificial_neural_network", "neural_networks.txt"),
        ("https://en.wikipedia.org/wiki/Python_(programming_language)", "python_lang.txt"),
        ("https://en.wikipedia.org/wiki/Machine_learning", "machine_learning.txt"),
        ("https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)", "transformers.txt"),
        ("https://docs.python.org/3/tutorial/controlflow.html", "python_control_flow.txt"),
        ("https://docs.python.org/3/tutorial/datastructures.html", "python_data_structures.txt"),
    ]
    
    for url, filename in TARGETS:
        out_path = os.path.join(raw_dir, filename)
        scrape_article(url, out_path)
        
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    run_spider(RAW_DIR)
