import json
import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import re

def parse_answer(text):
    """
    Naively extract answer from text. 
    Assumes format like '{final answer: 123}' or just looks for last number.
    This mimics the logic in evaluator.py but simplified for analysis.
    """
    # Try curly bracket format first
    match = re.search(r'\{final answer: (.*?)\}', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "N/A"

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze(file_path, target_agent_index=0, model_name='all-MiniLM-L6-v2'):
    print(f"Loading SBERT model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Loading data from {file_path}...")
    data = load_data(file_path)
    
    metrics = {
        'semantic_drift': [], # Distance between Target(t) and Target(t-1)
        'consensus_dist': [], # Distance between Target(t) and Centroid(Peers(t))
        'accuracy': []        # Target accuracy trend
    }
    
    # Identify Target Agent Name pattern
    # The keys in 'responses' are like "gsm8k_100__qwen...__Agent1"
    # We need to find the one corresponding to index 0, 1, etc.
    # We assume standard sorting or naming convention.
    
    for record in tqdm(data, desc="Analyzing questions"):
        # Helper to get sorting
        # Look at Round 0
        if '0' not in record: continue
        
        # Check if dual generation structure exists
        r0 = record['0']
        if 'variant' in r0['responses']:
            responses_map = r0['responses']['variant'] # Use Variant universe for Target
        else:
            responses_map = r0['responses'] # Fallback
            
        agent_names = sorted(list(responses_map.keys()))
        target_name = agent_names[target_agent_index]
        
        # Track per-round history for this question
        target_history = []
        
        # Iterate rounds
        # Sorting rounds '0', '1', '2'...
        rounds = sorted([k for k in record.keys() if k.isdigit()], key=int)
        
        for r_key in rounds:
            r_data = record[r_key]
            
            # Access responses
            if 'variant' in r_data['responses']:
                 resp_dict = r_data['responses']['variant']
            else:
                 resp_dict = r_data['responses']
                 
            target_text = resp_dict[target_name]
            target_history.append(target_text)
            
            # Compute Semantics
            target_emb = model.encode(target_text, convert_to_tensor=True)
            
            # 1. Self-Consistency Drift (vs previous round)
            if len(target_history) > 1:
                prev_text = target_history[-2]
                prev_emb = model.encode(prev_text, convert_to_tensor=True)
                sim = util.pytorch_cos_sim(target_emb, prev_emb).item()
                drift = 1.0 - sim
                metrics['semantic_drift'].append({'round': int(r_key), 'val': drift})
            
            # 2. Consensus Distance (vs Peers)
            peer_texts = [resp_dict[name] for name in agent_names if name != target_name]
            if peer_texts:
                peer_embs = model.encode(peer_texts, convert_to_tensor=True)
                centroid = peer_embs.mean(dim=0)
                sim_to_centroid = util.pytorch_cos_sim(target_emb, centroid).item()
                dist_to_centroid = 1.0 - sim_to_centroid
                metrics['consensus_dist'].append({'round': int(r_key), 'val': dist_to_centroid})

    # Aggregation
    print("\nResults Summary:")
    drift_by_round = {}
    for item in metrics['semantic_drift']:
        r = item['round']
        if r not in drift_by_round: drift_by_round[r] = []
        drift_by_round[r].append(item['val'])
        
    print("Self-Consistency Drift (1 - CosSimilarity) by Round:")
    for r in sorted(drift_by_round.keys()):
        print(f"  Round {r}: {np.mean(drift_by_round[r]):.4f} (+/- {np.std(drift_by_round[r]):.4f})")
        
    cons_by_round = {}
    for item in metrics['consensus_dist']:
        r = item['round']
        if r not in cons_by_round: cons_by_round[r] = []
        cons_by_round[r].append(item['val'])

    print("\nDistance to Peer Centroid (1 - CosSimilarity) by Round:")
    for r in sorted(cons_by_round.keys()):
        print(f"  Round {r}: {np.mean(cons_by_round[r]):.4f} (+/- {np.std(cons_by_round[r]):.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to .jsonl output file')
    args = parser.parse_args()
    
    analyze(args.file)
