import json
import glob
import os
import re
import argparse
import matplotlib.pyplot as plt

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


def parse_filename_metadata(basename):
    # Current format (temperature-free):
    # dataset_size__model_N=..._R=....jsonl
    # Legacy files with _TR/_TT/... are also supported.
    pattern = re.compile(r"([a-zA-Z0-9_]+)_(\d+)__(.+?)_N=.*?(?:_MNT=(\d+))?\.jsonl$")
    match = pattern.match(basename)
    if not match:
        return None

    tr_match = re.search(r"_TR=(-?\d+)", basename)

    return {
        "dataset": match.group(1),
        "data_size": int(match.group(2)),
        "model": match.group(3),
        "max_new_tokens": int(match.group(4)) if match.group(4) else 512,
        "legacy_target_round": int(tr_match.group(1)) if tr_match else None,
    }


def extract_and_save(dataset_filter=None, history_dir=HISTORY_DIR):
    results = []

    # Use recursive glob to find files in subdirectories, but exclude backup directories
    all_files = glob.glob(os.path.join(history_dir, "**", "*.jsonl"), recursive=True)
    files = [f for f in all_files if "previous_backup" not in f]
    print(f"Found {len(files)} JSONL files in {history_dir} (excluded {len(all_files) - len(files)} backup files)")

    for filename in files:
        basename = os.path.basename(filename)
        meta = parse_filename_metadata(basename)
        if meta is None:
            # Skip files that don't match the experiment pattern
            continue

        # Filter by dataset if specified
        if dataset_filter and meta["dataset"] != dataset_filter:
            continue

        print(f"Processing {basename} (dataset={meta['dataset']})...")

        total = 0
        correct_counts = {}  # key: round_num (int), value: count of correct answers

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
                    solutions = data

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
        for r in sorted(correct_counts.keys()):
            round_accuracies[str(r)] = correct_counts[r] / total

        record = {
            "dataset": meta["dataset"],
            "data_size": meta["data_size"],
            "model": meta["model"],
            "max_new_tokens": meta["max_new_tokens"],
            "legacy_target_round": meta["legacy_target_round"],
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

    # Group by dataset, model, data_size, max_new_tokens
    grouped_data = {}
    for res in results:
        dataset = res.get('dataset', 'unknown')
        model = res.get('model', 'unknown')
        data_size = res.get('data_size', 0)
        mnt = res.get('max_new_tokens', 512)

        group_key = (dataset, model, data_size, mnt)
        if group_key not in grouped_data:
            grouped_data[group_key] = []

        grouped_data[group_key].append(res)

    for (dataset, model_name, data_size, mnt), records in grouped_data.items():
        mnt_label = f", MNT={mnt}" if mnt != 512 else ""
        print(f"\nGenerating plot for dataset: {dataset}, model: {model_name}, size: {data_size}{mnt_label}")

        plt.figure(figsize=(10, 6))
        records.sort(key=lambda x: x['source_file'])

        for idx, res in enumerate(records):
            round_keys = sorted([int(k) for k in res['round_accuracies'].keys()])
            if not round_keys:
                continue

            round_accs = [res['round_accuracies'][str(k)] for k in round_keys]
            legacy_tr = res.get('legacy_target_round')
            if legacy_tr is None:
                label = f"run_{idx+1}"
            else:
                label = f"TR={legacy_tr}"
            plt.plot(round_keys, round_accs, marker='o', label=label)

        title_suffix = f" [MNT={mnt}]" if mnt != 512 else ""
        plt.title(f'{dataset.upper()} ({model_name}, N={data_size}{title_suffix}): Accuracy by Debate Round')
        plt.xlabel('Debate Round')
        plt.ylabel('Accuracy')
        plt.legend(title='Run')
        plt.grid(True)

        safe_model_name = model_name.replace('/', '_')
        mnt_suffix = f"_MNT={mnt}" if mnt != 512 else ""
        output_plot_path = os.path.join(history_dir, f'{dataset}_{safe_model_name}_N{data_size}_accuracy_by_round{mnt_suffix}.png')
        plt.savefig(output_plot_path)
        print(f"Saved plot to {output_plot_path}")
        plt.close()

if __name__ == "__main__":
    args = get_args()
    data = extract_and_save(dataset_filter=args.dataset, history_dir=args.history_dir)
    visualize(data, dataset_filter=args.dataset, history_dir=args.history_dir)
