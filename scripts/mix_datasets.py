import json
import random
import os
import argparse

def load_json(path):
    print(f"Loading {path}...")
    if not os.path.exists(path):
        print(f"Warning: File not found at {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_item(item, source_type):
    """
    Normalize item to ensure it has 'Question' and 'answer' keys.
    """
    normalized = item.copy()
    
    # Ensure 'Question' key exists
    if 'Question' not in normalized:
        if 'problem' in normalized:
            normalized['Question'] = normalized['problem']
        elif 'question' in normalized:
            normalized['Question'] = normalized['question']
    
    # Ensure 'answer' key exists
    if 'answer' not in normalized:
        if 'solution' in normalized:
            normalized['answer'] = normalized['solution']
            
    # Tag source for debugging clarity
    normalized['_source'] = source_type
    
    return normalized

def main():
    parser = argparse.ArgumentParser(description="Mix AIME and BrowserComp datasets.")
    parser.add_argument('--aime_test', type=str, default='/online1/ycsc_lijt1/lijt1/tyr/Search-o1-serper-sandbox/data/AIME/test.json', help='Path to AIME test.json')
    parser.add_argument('--aime_2024', type=str, default='/online1/ycsc_lijt1/lijt1/tyr/Search-o1-serper-sandbox/data/AIME/aime_2024.json', help='Path to AIME 2024.json')
    parser.add_argument('--browser_test', type=str, default='/online1/ycsc_lijt1/lijt1/tyr/Search-o1-serper-sandbox/data/BrowserComp/test.json', help='Path to BrowserComp test.json')
    parser.add_argument('--output', type=str, default='mixed_dataset.json', help='Output JSON path')
    parser.add_argument('--count', type=int, default=30, help='Number of items to sample from EACH source')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Adjust paths for local environment if needed (for my verification)
    # The user provided absolute paths, but locally they might be relative.
    # I'll check if the absolute path exists, if not, try to map to local 'd:\...' or relative 'data/...'
    
    def resolve_path(p):
        if os.path.exists(p):
            return p
        # Try finding in local data dir matching the end pattern
        if 'AIME/test.json' in p: return 'data/AIME/test.json'
        if 'AIME/aime_2024.json' in p: return 'data/AIME/aime_2024.json'
        if 'BrowserComp/test.json' in p: return 'data/BrowserComp/test.json'
        return p

    path1 = resolve_path(args.aime_test)
    path2 = resolve_path(args.aime_2024)
    path3 = resolve_path(args.browser_test)
    
    data1 = load_json(path1)
    data2 = load_json(path2)
    data3 = load_json(path3)
    
    print(f"Loaded counts: AIME Test ({len(data1)}), AIME 2024 ({len(data2)}), BrowserComp ({len(data3)})")
    
    # Sample
    sample1 = random.sample(data1, min(args.count, len(data1)))
    sample2 = random.sample(data2, min(args.count, len(data2)))
    sample3 = random.sample(data3, min(args.count, len(data3)))
    
    # Normalize
    processed_data = []
    for item in sample1:
        processed_data.append(normalize_item(item, 'AIME_Test'))
    for item in sample2:
        processed_data.append(normalize_item(item, 'AIME_2024'))
    for item in sample3:
        processed_data.append(normalize_item(item, 'BrowserComp'))
        
    # Shuffle combined data
    random.shuffle(processed_data)
    
    # Save
    print(f"Writing {len(processed_data)} items to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    main()
