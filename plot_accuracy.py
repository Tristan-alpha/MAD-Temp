
import os
import glob
import re
import json
import matplotlib.pyplot as plt

def extract_temperature(filename):
    # Regex to capture T=...
    # Uses \d+(\.\d+)? to capture integer or float, stopping before the file extension dot or other characters
    match = re.search(r"T=(\d+(?:\.\d+)?)", filename)
    if match:
        return float(match.group(1))
    return None

def main():
    files = glob.glob("out/history/*.jsonl")
    data_by_round = {} # round_idx -> list of (T, accuracy)

    for filepath in files:
        filename = os.path.basename(filepath)
        T = extract_temperature(filename)
        
        if T is None:
            print(f"Skipping {filename}: No temperature found in filename.")
            continue
            
        try:
            cumulative_acc_line = None
            with open(filepath, 'r') as f:
                # Read all lines and find the last cumulative_accuracy
                for line in f:
                    if '"cumulative_accuracy"' in line:
                        cumulative_acc_line = line
            
            if cumulative_acc_line:
                # Parse the line
                # It should be a JSON object like {"cumulative_accuracy": [0.96, ...]}
                # Sometimes it might be embedded or have other keys, but json.loads should handle valid json line
                data = json.loads(cumulative_acc_line)
                if "cumulative_accuracy" in data:
                    accuracies = data["cumulative_accuracy"]
                    
                    for round_idx, acc in enumerate(accuracies):
                        if round_idx not in data_by_round:
                            data_by_round[round_idx] = []
                        data_by_round[round_idx].append((T, acc))
                else:
                    print(f"Skipping {filename}: 'cumulative_accuracy' key not found in the line.")

            else:
                print(f"Skipping {filename}: No 'cumulative_accuracy' found.")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Sort rounds to ensure legend order
    sorted_rounds = sorted(data_by_round.keys())
    
    for round_idx in sorted_rounds:
        points = data_by_round[round_idx]
        # Sort by Temperature
        points.sort(key=lambda x: x[0])
        
        temps = [p[0] for p in points]
        accs = [p[1] for p in points]
        
        plt.plot(temps, accs, marker='o', label=f'Round {round_idx}')

    plt.xlabel('Temperature')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Temperature per Round')
    plt.legend()
    plt.grid(True)
    
    output_image = "temperature_accuracy_plot.png"
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    main()
