import json
import glob
import os
import re
import matplotlib.pyplot as plt
# import pandas as pd # Removed dependency
import numpy as np

# Configuration
HISTORY_DIR = "/export/home3/dazhou/debate-or-vote/out/history"
OUTPUT_JSONL = os.path.join(HISTORY_DIR, "accuracy_summary.jsonl")

def extract_and_save():
    # Pattern to match filenames: e.g., gsm8k_50__qwen2.5-7b_N=3_R=3_TR=1_TT=0.1.jsonl
    # We are looking for TR (Target Round) and TT (Target Temperature)
    pattern = re.compile(r".*_TR=(\d+)_TT=([\d\.]+)\.jsonl")
    
    results = []
    
    files = glob.glob(os.path.join(HISTORY_DIR, "*.jsonl"))
    print(f"Found {len(files)} JSONL files in {HISTORY_DIR}")
    
    for filename in files:
        basename = os.path.basename(filename)
        match = pattern.match(basename)
        if not match:
            # Skip files that don't match the experiment pattern
            continue
            
        tr = int(match.group(1))
        tt = float(match.group(2))
        
        print(f"Processing {basename} (TR={tr}, TT={tt})...")
        
        total = 0
        correct_counts = {} # key: round_num (int), value: count of correct answers
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                        
                    total += 1
                    # solutions = data.get("solutions", {}) # OLD
                    solutions = data # Top level seems to be the rounds
                    
                    # Iterate over rounds in the solution
                    for str_round, round_data in solutions.items():
                        try:
                            round_num = int(str_round)
                            # debate_answer_iscorr is the correctness of the consensus/debate answer for that round
                            is_correct = round_data.get("debate_answer_iscorr", False)
                            
                            if is_correct:
                                correct_counts[round_num] = correct_counts.get(round_num, 0) + 1
                            else:
                                # Ensure the key exists even if count is 0
                                if round_num not in correct_counts:
                                    correct_counts[round_num] = 0
                        except ValueError:
                            continue
                            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        if total == 0:
            print(f"Warning: {basename} is empty or has no valid data.")
            continue
            
        # Calculate accuracy for each round
        round_accuracies = {}
        # We sort keys to ensure order
        for r in sorted(correct_counts.keys()):
            round_accuracies[str(r)] = correct_counts[r] / total
            
        record = {
            "target_round": tr,
            "temperature": tt,
            "round_accuracies": round_accuracies,
            "total_samples": total,
            "source_file": basename
        }
        results.append(record)
        
    # Write summary to JSONL
    print(f"Writing {len(results)} records to {OUTPUT_JSONL}")
    with open(OUTPUT_JSONL, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    return results

def visualize(results):
    if not results:
        print("No results to visualize.")
        return

    # Group by target_round
    # Dict structure: { tr_value: [record1, record2, ...] }
    grouped_by_tr = {}
    for res in results:
        tr = res['target_round']
        if tr not in grouped_by_tr:
            grouped_by_tr[tr] = []
        grouped_by_tr[tr].append(res)
    
    # Sort target rounds
    target_rounds = sorted(grouped_by_tr.keys())
    
    for tr in target_rounds:
        records = grouped_by_tr[tr]
        # Sort records by temperature
        records.sort(key=lambda x: x['temperature'])
        
        plt.figure(figsize=(10, 6))
        
        # Identify all rounds present in this subset
        all_rounds = set()
        for res in records:
            all_rounds.update(map(int, res['round_accuracies'].keys()))
        
        sorted_rounds = sorted(list(all_rounds))
        
        # Plot a line for each round
        for r in sorted_rounds:
            temps = []
            accs = []
            
            for res in records:
                t = res['temperature']
                a = res['round_accuracies'].get(str(r), None)
                
                if a is not None:
                    temps.append(t)
                    accs.append(a)
            
            if temps:
                plt.plot(temps, accs, marker='o', label=f'Round {r}')
        
        plt.title(f'Accuracy vs Temperature (Target Round {tr})')
        plt.xlabel('Temperature')
        plt.ylabel('Accuracy')
        plt.legend(title="Round")
        plt.grid(True)
        
        output_plot_path = os.path.join(HISTORY_DIR, f'accuracy_plot_TR_{tr}.png')
        plt.savefig(output_plot_path)
        print(f"Saved plot to {output_plot_path}")
        plt.close()

if __name__ == "__main__":
    data = extract_and_save()
    visualize(data)
