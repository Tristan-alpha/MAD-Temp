#!/usr/bin/env python3
"""
GPU Monitor Script - View GPU memory usage across all partitions in a Slurm cluster.
Usage: python gpu_monitor.py [--html] [--timeout SECONDS]
"""
# watch -n 1 -c "python scripts/gpu_monitor.py --partitions NA100q"

import subprocess
import re
import base64

import argparse
from collections import defaultdict
from datetime import datetime
import os

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def get_partitions_and_nodes():
    """Get all partitions and their nodes from sinfo."""
    result = subprocess.run(
        ['sinfo', '-h', '-o', '%P %T %N'],
        capture_output=True, text=True
    )
    
    partition_nodes = defaultdict(list)
    partition_states = defaultdict(dict)
    
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 3:
            partition = parts[0].rstrip('*')  # Remove default marker
            state = parts[1]
            nodelist = parts[2]
            
            # Skip unavailable states
            if state in ['down', 'down*', 'drain', 'drain*', 'draining']:
                continue
            
            # Expand node list (e.g., node[01-03] -> node01, node02, node03)
            expanded = expand_nodelist(nodelist)
            for node in expanded:
                if node not in partition_nodes[partition]:
                    partition_nodes[partition].append(node)
                    partition_states[partition][node] = state
    
    return partition_nodes, partition_states

def expand_nodelist(nodelist):
    """Expand Slurm nodelist format to individual node names."""
    if not nodelist or nodelist == '(null)':
        return []
    
    result = subprocess.run(
        ['scontrol', 'show', 'hostnames', nodelist],
        capture_output=True, text=True
    )
    return result.stdout.strip().split('\n') if result.stdout.strip() else []

def get_gpu_info_via_srun(partition, node, timeout=15):
    """Get GPU info from a node using srun."""
    # Remote python script to execute on the compute node
    remote_code = r"""
import subprocess, csv, io, re, sys

def get_job_id(pid):
    try:
        with open(f"/proc/{pid}/cgroup", "r") as f:
            content = f.read()
            match = re.search(r"job_(\d+)", content)
            if match: return match.group(1)
    except: pass
    return ""

def main():
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0: return

        gpus = []
        for row in csv.reader(io.StringIO(res.stdout)):
            if len(row) >= 8:
                try: u = int(row[5])
                except: u = 0
                try: p_draw = float(row[6])
                except: p_draw = 0.0
                try: p_limit = float(row[7])
                except: p_limit = 0.0
                
                gpus.append({
                    "id": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip(),
                    "used": int(row[3]), "total": int(row[4]), "util": u,
                    "p_draw": p_draw, "p_limit": p_limit,
                    "jobs": set()
                })

        cmd_proc = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
        res_proc = subprocess.run(cmd_proc, capture_output=True, text=True)
        if res_proc.returncode == 0:
            for row in csv.reader(io.StringIO(res_proc.stdout)):
                if len(row) >= 2:
                    uuid, pid = row[0].strip(), row[1].strip()
                    jid = get_job_id(pid)
                    if jid:
                        for g in gpus:
                            if g["uuid"] == uuid: g["jobs"].add(jid)

        for g in gpus:
            jstr = " ".join(sorted(g["jobs"])) if g["jobs"] else "Idle"
            print(f"{g['id']},{g['name']},{g['used']},{g['total']},{g['util']},{g['p_draw']},{g['p_limit']},{jstr}")
    except: pass

if __name__=="__main__": main()
"""
    try:
        # Encode script to avoid escaping issues
        encoded_script = base64.b64encode(remote_code.encode('utf-8')).decode('utf-8')
        
        cmd = [
            'srun', '-p', partition, '-w', node,
            '--time=00:01:00',
            'python3', '-c', f"import base64; exec(base64.b64decode('{encoded_script}').decode('utf-8'))"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            return None, f"srun failed: {result.stderr.strip()[:50]}"
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            # Logic to skip lines that are NOT csv
            if ',' not in line: continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 8:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used': int(parts[2]),
                    'memory_total': int(parts[3]),
                    'utilization': int(parts[4]),
                    'power_draw': float(parts[5]),
                    'power_limit': float(parts[6]),
                    'jobs': parts[7]
                })
        
        return gpus, None
    
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)[:50]

def get_memory_bar(used, total, width=20):
    """Generate a text-based progress bar for memory usage."""
    if total == 0:
        return "[N/A]"
    
    ratio = used / total
    filled = int(width * ratio)
    empty = width - filled
    
    # Color based on usage
    if ratio < 0.5:
        color = Colors.GREEN
    elif ratio < 0.8:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    
    bar = f"{color}{'█' * filled}{'░' * empty}{Colors.END}"
    
    return f"[{bar}]"

def print_results(all_data):
    """Print results in a formatted table."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}  GPU Memory Usage Monitor - {timestamp}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}\n")
    
    for partition in sorted(all_data.keys()):
        nodes = all_data[partition]
        
        # Calculate partition summary
        total_gpus = 0
        total_used = 0
        total_memory = 0
        
        for node, info in nodes.items():
            if info['gpus']:
                total_gpus += len(info['gpus'])
                for gpu in info['gpus']:
                    total_used += gpu['memory_used']
                    total_memory += gpu['memory_total']
        
        # Print partition header
        print(f"{Colors.BOLD}{Colors.BLUE}┌{'─' * 78}┐{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}│ Partition: {partition:<20} │ Total GPUs: {total_gpus:<5} │ Memory: {total_used:>6}/{total_memory:>6} MiB │{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}└{'─' * 78}┘{Colors.END}")
        
        for node, info in sorted(nodes.items()):
            if info['error']:
                print(f"  {Colors.YELLOW}├─ {node}: ⚠ {info['error']}{Colors.END}")
                continue
            
            if not info['gpus']:
                print(f"  {Colors.YELLOW}├─ {node}: No GPUs found{Colors.END}")
                continue
            
            print(f"  {Colors.CYAN}├─ {node} ({len(info['gpus'])} GPUs):{Colors.END}")
            
            for gpu in info['gpus']:
                bar = get_memory_bar(gpu['memory_used'], gpu['memory_total'])
                
                # Calculate Power Percentage
                p_draw = gpu.get('power_draw', 0.0)
                p_limit = gpu.get('power_limit', 0.0)
                p_pct = (p_draw / p_limit * 100) if p_limit > 0 else 0.0
                power_str = f"Pwr: {p_pct:5.1f}%"
                
                job_id = gpu.get('jobs', 'Unknown')
                job_str = f"Job: {job_id}"
                
                print(f"  │  GPU {gpu['index']}: {gpu['name']:<25} {gpu['memory_used']:>6}/{gpu['memory_total']:>6} MiB {bar} {power_str} {job_str}")
        
        print()

def generate_html_report(all_data, output_path):
    """Generate an HTML visualization report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Monitor - {timestamp}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: #00d4ff;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }}
        .timestamp {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .partition {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .partition-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .partition-name {{
            font-size: 24px;
            font-weight: bold;
            color: #00d4ff;
        }}
        .partition-stats {{
            display: flex;
            gap: 20px;
        }}
        .stat {{
            background: rgba(0, 212, 255, 0.1);
            padding: 8px 15px;
            border-radius: 8px;
            font-size: 14px;
        }}
        .stat-value {{ color: #00d4ff; font-weight: bold; }}
        .node {{
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        .node-header {{
            font-size: 16px;
            font-weight: bold;
            color: #4fc3f7;
            margin-bottom: 12px;
        }}
        .gpu-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 12px;
        }}
        .gpu-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 12px;
            border-left: 4px solid #4fc3f7;
        }}
        .gpu-name {{
            font-size: 12px;
            color: #aaa;
            margin-bottom: 8px;
        }}
        .gpu-index {{
            font-weight: bold;
            color: #fff;
            margin-right: 8px;
        }}
        .memory-info {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            font-size: 14px;
        }}
        .memory-bar-container {{
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }}
        .memory-bar {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        .memory-bar.low {{ background: linear-gradient(90deg, #00c853, #69f0ae); }}
        .memory-bar.medium {{ background: linear-gradient(90deg, #ffc107, #ffca28); }}
        .memory-bar.high {{ background: linear-gradient(90deg, #ff5252, #ff1744); }}
        .memory-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 11px;
            font-weight: bold;
            text-shadow: 0 0 3px rgba(0, 0, 0, 0.8);
        }}
        .utilization {{
            font-size: 12px;
            color: #888;
            margin-top: 6px;
        }}
        .error {{
            color: #ff9800;
            font-style: italic;
            padding: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: rgba(0, 212, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(0, 212, 255, 0.3);
        }}
        .summary-value {{
            font-size: 36px;
            font-weight: bold;
            color: #00d4ff;
        }}
        .summary-label {{
            font-size: 14px;
            color: #888;
            margin-top: 5px;
        }}
        .refresh-btn {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #00d4ff;
            color: #1a1a2e;
            border: none;
            padding: 15px 25px;
            border-radius: 30px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        }}
        .refresh-btn:hover {{
            background: #00b8e6;
            transform: scale(1.05);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🖥️ GPU Memory Usage Monitor</h1>
        <p class="timestamp">Last updated: {timestamp}</p>
'''
    
    # Calculate overall summary
    total_partitions = len(all_data)
    total_nodes = sum(len(nodes) for nodes in all_data.values())
    total_gpus = 0
    total_memory_used = 0
    total_memory_all = 0
    
    for partition, nodes in all_data.items():
        for node, info in nodes.items():
            if info['gpus']:
                total_gpus += len(info['gpus'])
                for gpu in info['gpus']:
                    total_memory_used += gpu['memory_used']
                    total_memory_all += gpu['memory_total']
    
    overall_usage = (total_memory_used / total_memory_all * 100) if total_memory_all > 0 else 0
    
    html_content += f'''
        <div class="summary">
            <div class="summary-card">
                <div class="summary-value">{total_partitions}</div>
                <div class="summary-label">Partitions</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{total_nodes}</div>
                <div class="summary-label">Nodes</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{total_gpus}</div>
                <div class="summary-label">Total GPUs</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{overall_usage:.1f}%</div>
                <div class="summary-label">Overall Memory Usage</div>
            </div>
        </div>
'''
    
    # Generate partition sections
    for partition in sorted(all_data.keys()):
        nodes = all_data[partition]
        
        # Partition stats
        p_gpus = 0
        p_used = 0
        p_total = 0
        
        for node, info in nodes.items():
            if info['gpus']:
                p_gpus += len(info['gpus'])
                for gpu in info['gpus']:
                    p_used += gpu['memory_used']
                    p_total += gpu['memory_total']
        
        p_usage = (p_used / p_total * 100) if p_total > 0 else 0
        
        html_content += f'''
        <div class="partition">
            <div class="partition-header">
                <span class="partition-name">📦 {partition}</span>
                <div class="partition-stats">
                    <span class="stat">GPUs: <span class="stat-value">{p_gpus}</span></span>
                    <span class="stat">Memory: <span class="stat-value">{p_used:,}/{p_total:,} MiB</span></span>
                    <span class="stat">Usage: <span class="stat-value">{p_usage:.1f}%</span></span>
                </div>
            </div>
'''
        
        for node, info in sorted(nodes.items()):
            if info['error']:
                html_content += f'''
            <div class="node">
                <div class="node-header">🖧 {node}</div>
                <p class="error">⚠️ {info['error']}</p>
            </div>
'''
                continue
            
            if not info['gpus']:
                html_content += f'''
            <div class="node">
                <div class="node-header">🖧 {node}</div>
                <p class="error">No GPUs detected</p>
            </div>
'''
                continue
            
            html_content += f'''
            <div class="node">
                <div class="node-header">🖧 {node} ({len(info['gpus'])} GPUs)</div>
                <div class="gpu-grid">
'''
            
            for gpu in info['gpus']:
                ratio = gpu['memory_used'] / gpu['memory_total'] if gpu['memory_total'] > 0 else 0
                percentage = ratio * 100
                
                if ratio < 0.5:
                    bar_class = 'low'
                elif ratio < 0.8:
                    bar_class = 'medium'
                else:
                    bar_class = 'high'
                
                html_content += f'''
                    <div class="gpu-card">
                        <div class="gpu-name"><span class="gpu-index">GPU {gpu['index']}</span>{gpu['name']}</div>
                        <div class="memory-info">
                            <span>{gpu['memory_used']:,} / {gpu['memory_total']:,} MiB</span>
                            <span>{percentage:.1f}%</span>
                        </div>
                        <div class="memory-bar-container">
                            <div class="memory-bar {bar_class}" style="width: {percentage}%"></div>
                            <span class="memory-text">{gpu['memory_used']:,} MiB</span>
                        </div>
                        <div class="utilization">GPU Utilization: {gpu['utilization']}%</div>
                    </div>
'''
            
            html_content += '''
                </div>
            </div>
'''
        
        html_content += '''
        </div>
'''
    
    html_content += '''
        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
    </div>
</body>
</html>
'''
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n{Colors.GREEN}✓ HTML report saved to: {output_path}{Colors.END}")

def main():
    parser = argparse.ArgumentParser(description='Monitor GPU memory usage across Slurm partitions')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--output', type=str, default='gpu_monitor.html', help='HTML output file path')
    parser.add_argument('--timeout', type=int, default=20, help='Timeout for srun commands (seconds)')
    parser.add_argument('--partitions', type=str, nargs='+', help='Only check specific partitions')
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Fetching partition and node information...{Colors.END}")
    partition_nodes, partition_states = get_partitions_and_nodes()
    
    if args.partitions:
        partition_nodes = {p: n for p, n in partition_nodes.items() if p in args.partitions}
    
    if not partition_nodes:
        print(f"{Colors.RED}No available partitions found.{Colors.END}")
        return
    
    print(f"Found {len(partition_nodes)} partitions with available nodes.\n")
    
    all_data = {}
    
    for partition in sorted(partition_nodes.keys()):
        nodes = partition_nodes[partition]
        print(f"{Colors.CYAN}Checking partition: {partition} ({len(nodes)} nodes)...{Colors.END}")
        
        all_data[partition] = {}
        
        for node in nodes:
            print(f"  - Querying {node}...", end=' ', flush=True)
            gpus, error = get_gpu_info_via_srun(partition, node, args.timeout)
            
            if error:
                print(f"{Colors.YELLOW}⚠ {error}{Colors.END}")
            else:
                print(f"{Colors.GREEN}✓ {len(gpus)} GPUs{Colors.END}")
            
            all_data[partition][node] = {
                'gpus': gpus,
                'error': error,
                'state': partition_states[partition].get(node, 'unknown')
            }
    
    # Print results
    print_results(all_data)
    
    # Generate HTML if requested
    if args.html:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.path.dirname(__file__), '..', output_path)
        generate_html_report(all_data, output_path)

if __name__ == '__main__':
    main()
