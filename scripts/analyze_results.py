import json
import glob
import os
import re
import argparse
import matplotlib.pyplot as plt
# import pandas as pd # Removed dependency
import numpy as np

# Configuration
HISTORY_DIR = "/export/home3/dazhou/debate-or-vote/out/history"

def get_args():
    parser = argparse.ArgumentParser(description="Analyze experiment results and generate accuracy plots")
    parser.add_argument('--dataset', type=str, default=None, 
                        choices=['gsm8k', 'formal_logic', 'pro_medicine', 'arithmetics', 'csqa', 'hellaswag', 'hh_rlhf'],
                        help="Dataset to analyze. If not specified, all datasets will be analyzed separately.")
    parser.add_argument('--history_dir', type=str, default=HISTORY_DIR,
                        help="Directory containing the experiment history files")
    return parser.parse_args()

def extract_and_save(dataset_filter=None, history_dir=HISTORY_DIR):
    # Pattern to match filenames: e.g., gsm8k_50__qwen2.5-7b_N=3_R=3_TR=1_TT=0.1.jsonl
    # Also supports optional _MNT=8192 suffix: gsm8k_500__qwen3-4b_N=3_R=3_TR=2_TT=0.0_MNT=8192.jsonl
    pattern = re.compile(r"([a-zA-Z0-9_]+)_(\d+)__(.+?)_N=.*_TR=(\d+)_TT=([\d\.]+)(?:_MNT=(\d+))?\.jsonl")
    
    results = []
    
    # Use recursive glob to find files in subdirectories, but exclude backup directories
    all_files = glob.glob(os.path.join(history_dir, "**", "*.jsonl"), recursive=True)
    files = [f for f in all_files if "previous_backup" not in f]
    print(f"Found {len(files)} JSONL files in {history_dir} (excluded {len(all_files) - len(files)} backup files)")
    
    for filename in files:
        basename = os.path.basename(filename)
        match = pattern.match(basename)
        if not match:
            # Skip files that don't match the experiment pattern
            continue
        
        dataset_name = match.group(1)
        data_size = int(match.group(2))
        model_name = match.group(3)
        tr = int(match.group(4))
        tt = float(match.group(5))
        mnt = int(match.group(6)) if match.group(6) else 512  # default max_new_tokens
        
        # Filter by dataset if specified
        if dataset_filter and dataset_name != dataset_filter:
            continue
            
        print(f"Processing {basename} (dataset={dataset_name}, TR={tr}, TT={tt})...")
        
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
            "dataset": dataset_name,
            "data_size": data_size,
            "model": model_name,
            "target_round": tr,
            "temperature": tt,
            "max_new_tokens": mnt,
            "round_accuracies": round_accuracies,
            "total_samples": total,
            "source_file": basename
        }
        results.append(record)
        
    # Write summary to JSONL
    output_jsonl = os.path.join(history_dir, f"accuracy_summary{'_' + dataset_filter if dataset_filter else ''}.jsonl")
    print(f"Writing {len(results)} records to {output_jsonl}")
    with open(output_jsonl, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    return results

def visualize(results, dataset_filter=None, history_dir=HISTORY_DIR):
    if not results:
        print("No results to visualize.")
        return

    # Group by dataset, model, data_size, max_new_tokens, then target_round
    grouped_data = {}
    for res in results:
        dataset = res.get('dataset', 'unknown')
        model = res.get('model', 'unknown')
        data_size = res.get('data_size', 0)
        mnt = res.get('max_new_tokens', 512)
        tr = res['target_round']
        
        group_key = (dataset, model, data_size, mnt)
        if group_key not in grouped_data:
            grouped_data[group_key] = {}
        if tr not in grouped_data[group_key]:
            grouped_data[group_key][tr] = []
        
        grouped_data[group_key][tr].append(res)
    
    for (dataset, model_name, data_size, mnt), grouped_by_tr in grouped_data.items():
        mnt_label = f", MNT={mnt}" if mnt != 512 else ""
        print(f"\nGenerating plots for dataset: {dataset}, model: {model_name}, size: {data_size}{mnt_label}")
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
            
            title_suffix = f" [MNT={mnt}]" if mnt != 512 else ""
            plt.title(f'{dataset.upper()} ({model_name}, N={data_size}{title_suffix}): Accuracy vs Temperature (Target Round {tr})')
            plt.xlabel('Temperature')
            plt.ylabel('Accuracy')
            plt.legend(title="Round")
            plt.grid(True)
            
            safe_model_name = model_name.replace('/', '_')
            mnt_suffix = f"_MNT={mnt}" if mnt != 512 else ""
            output_plot_path = os.path.join(history_dir, f'{dataset}_{safe_model_name}_N{data_size}_accuracy_plot_TR_{tr}{mnt_suffix}.png')
            plt.savefig(output_plot_path)
            print(f"Saved plot to {output_plot_path}")
            plt.close()

if __name__ == "__main__":
    args = get_args()
    data = extract_and_save(dataset_filter=args.dataset, history_dir=args.history_dir)
    visualize(data, dataset_filter=args.dataset, history_dir=args.history_dir)
