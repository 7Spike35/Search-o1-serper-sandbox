import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize

# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)

def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        sentences = sent_tokenize(full_text)

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

def extract_pdf_text(url):
    """
    Extract text from a PDF.
    """
    try:
        response = session.get(url, timeout=20)
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        cleaned_text = ' '.join(full_text.split()[:600])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_url(url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None):
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.
    """
    try:
        if use_jina:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            response = session.get(url, timeout=20)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type:
                return extract_pdf_text(url)
            try:
                soup = BeautifulSoup(response.text, 'lxml')
            except Exception:
                # print("lxml parser not found or failed, falling back to html.parser")
                soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            return text[:8000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def fetch_page_content(urls, max_workers=4, use_jina=False, jina_api_key=None, snippets: Optional[dict] = None):
    """
    Concurrently fetch content from multiple URLs.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, jina_api_key, snippets.get(url) if snippets else None): url
            for url in urls
        }
        for future in concurrent.futures.as_completed(futures): # removed tqdm to reduce clutter if called frequently, or can keep it
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            time.sleep(0.2)
    return results

def serper_web_search(query, subscription_key, endpoint="https://google.serper.dev/search", market='en-US', language='en', timeout=20):
    """
    Perform a search using the Serper API.
    Note: 'endpoint' defaults to Serper's endpoint, but kept as arg for compatibility signature if needed.
    """
    url = "https://google.serper.dev/search"
    
    payload = json.dumps({
        "q": query,
        "gl": market.split('-')[-1] if '-' in market else 'us', # simple country code extraction
        "hl": language
    })
    
    headers = {
        'X-API-KEY': subscription_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Timeout:
        print(f"Serper Search request timed out ({timeout} seconds) for query: {query}")
        return {}
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during Serper Search request: {e}")
        return {}

def extract_relevant_info(search_results):
    """
    Extract relevant information from Serper search results.
    """
    useful_info = []
    
    if 'organic' in search_results:
        for id, result in enumerate(search_results['organic']):
            info = {
                'id': id + 1,
                'title': result.get('title', ''),
                'url': result.get('link', ''),
                'site_name': result.get('source', ''), # Serper sometimes has source
                'date': result.get('date', ''),
                'snippet': result.get('snippet', ''),
                'context': ''
            }
            useful_info.append(info)
    
    return useful_info

if __name__ == "__main__":
    # Example usage
    query = "Structure of dimethyl fumarate"
    SERPER_API_KEY = "YOUR_SERPER_API_KEY"
    
    if SERPER_API_KEY == "YOUR_SERPER_API_KEY":
        print("Please set your SERPER_API_KEY to test.")
    else:
        print("Performing Serper Web Search...")
        search_results = serper_web_search(query, SERPER_API_KEY)
        
        print("Extracting relevant information from search results...")
        extracted_info = extract_relevant_info(search_results)
        print(json.dumps(extracted_info, indent=2, ensure_ascii=False))
