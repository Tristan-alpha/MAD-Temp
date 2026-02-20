
import json
import glob
import os
import re

HISTORY_DIR = "/export/home3/dazhou/debate-or-vote/out/history"

def load_data():
    files = glob.glob(os.path.join(HISTORY_DIR, "**", "gsm8k_50*.jsonl"), recursive=True)
    all_data = []
    
    pattern = re.compile(r".*_TR=(\d+)_TT=([\d\.]+)\.jsonl")
    
    for f in files:
        basename = os.path.basename(f)
        match = pattern.match(basename)
        if not match: continue
        
        tr = int(match.group(1))
        tt = float(match.group(2))
        
        with open(f, 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                if not line.strip(): continue
                record = json.loads(line)
                
                # Determine correctness of final consensus
                is_correct = record.get('3', {}).get('debate_answer_iscorr', False)
                
                # Answer extraction
                answer = record.get('0', {}).get('answer', 'N/A')
                
                # Process rounds
                rounds_info = {}
                for r_key in ['0', '1', '2', '3']:
                    if r_key in record:
                        r_data = record[r_key]
                        responses = r_data.get('responses', {})
                        # Normalize agent keys to simple "Agent 1", "Agent 2" etc
                        agents_resps = {}
                        for k, v in responses.items():
                            # key like "gsm8k_50__qwen2.5-7b__None__Agent1"
                            short_k = k.split("__")[-1] # Agent1
                            agents_resps[short_k] = v
                        
                        final_answers = r_data.get('final_answers', [])
                        
                        rounds_info[r_key] = {
                            "responses": agents_resps,
                            "final_answers": final_answers,
                            "consensus": r_data.get('debate_answer')
                        }

                entry = {
                    "id": idx,
                    "file": basename,
                    "TR": tr,
                    "TT": tt,
                    "is_correct": is_correct,
                    "answer": answer,
                    "rounds": rounds_info
                }
                all_data.append(entry)
                
    return all_data

def generate_html(data):
    # Sort data for easier navigation
    data.sort(key=lambda x: (x['TR'], x['TT'], x['id']))
    
    # JSON dump for JS
    data_json = json.dumps(data)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Debate History Visualizer</title>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
        .controls {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; gap: 20px; flex-wrap: wrap; }}
        select {{ padding: 8px; border-radius: 4px; border: 1px solid #ccc; }}
        .container {{ display: flex; gap: 20px; }}
        .sidebar {{ width: 250px; background: white; padding: 10px; height: 80vh; overflow-y: auto; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .content {{ flex: 1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-height: 80vh; }}
        .item-row {{ padding: 8px; border-bottom: 1px solid #eee; cursor: pointer; }}
        .item-row:hover {{ background: #f5f5f5; }}
        .item-row.active {{ background: #e3f2fd; font-weight: bold; }}
        .status-dot {{ height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }}
        .correct {{ background-color: #4caf50; }}
        .incorrect {{ background-color: #f44336; }}
        
        .round-box {{ border: 1px solid #ddd; margin-bottom: 15px; padding: 10px; border-radius: 6px; }}
        .round-header {{ font-weight: bold; margin-bottom: 10px; background: #f9f9f9; padding: 5px; }}
        .agent-response {{ margin-bottom: 10px; border-left: 3px solid #007bff; padding-left: 10px; }}
        .agent-name {{ font-weight: bold; color: #0056b3; font-size: 0.9em; }}
        .response-text {{ font-family: monospace; white-space: pre-wrap; background: #fafafa; padding: 10px; font-size: 0.9em; max-height: 300px; overflow-y: auto; }}
        .final-parsed {{ color: #d63384; font-weight: bold; font-size: 0.9em; margin-top: 5px; }}
        
        .filters label {{ font-weight: bold; margin-right: 5px; }}
    </style>
</head>
<body>

<h1>Debate History Visualizer</h1>

<div class="controls filters">
    <div>
        <label>Target Round:</label>
        <select id="trFilter" onchange="filterList()">
            <option value="all">All</option>
            <option value="1">1</option>
            <option value="2">2</option>
        </select>
    </div>
    <div>
        <label>Temperature:</label>
        <select id="ttFilter" onchange="filterList()">
            <option value="all">All</option>
            <!-- Options populated by JS -->
        </select>
    </div>
    <div>
        <label>Status:</label>
        <select id="statusFilter" onchange="filterList()">
            <option value="all">All</option>
            <option value="correct">Correct</option>
            <option value="incorrect">Incorrect</option>
        </select>
    </div>
</div>

<div class="container">
    <div class="sidebar" id="itemList">
        <!-- List Items -->
    </div>
    <div class="content" id="detailView">
        <p>Select an item to view details.</p>
    </div>
</div>

<script>
    const allData = {data_json};
    let currentData = allData;
    
    // Populate TT filter
    const uniqueTT = [...new Set(allData.map(d => d.TT))].sort((a,b)=>a-b);
    const ttSelect = document.getElementById('ttFilter');
    uniqueTT.forEach(tt => {{
        let opt = document.createElement('option');
        opt.value = tt;
        opt.innerText = tt;
        ttSelect.appendChild(opt);
    }});

    function filterList() {{
        const trVal = document.getElementById('trFilter').value;
        const ttVal = document.getElementById('ttFilter').value;
        const statusVal = document.getElementById('statusFilter').value;
        
        currentData = allData.filter(d => {{
            if (trVal !== 'all' && d.TR != trVal) return false;
            if (ttVal !== 'all' && d.TT != ttVal) return false;
            if (statusVal === 'correct' && !d.is_correct) return false;
            if (statusVal === 'incorrect' && d.is_correct) return false;
            return true;
        }});
        
        renderList();
    }}
    
    function renderList() {{
        const list = document.getElementById('itemList');
        list.innerHTML = '';
        currentData.forEach((d, index) => {{
            const div = document.createElement('div');
            div.className = 'item-row';
            div.onclick = () => showDetail(d, div);
            div.innerHTML = `
                <span class="status-dot ${{d.is_correct ? 'correct' : 'incorrect'}}"></span>
                ID ${{d.id}} | TR=${{d.TR}} T=${{d.TT}}
            `;
            list.appendChild(div);
        }});
    }}
    
    function showDetail(d, element) {{
        // Highlight active
        document.querySelectorAll('.item-row').forEach(el => el.classList.remove('active'));
        element.classList.add('active');
        
        const view = document.getElementById('detailView');
        let html = `<h2>ID ${{d.id}} (Ans: ${{d.answer}})</h2>`;
        html += `<p><strong>File:</strong> ${{d.file}} <br> <strong>Result:</strong> <span style="color:${{d.is_correct?'green':'red'}}">${{d.is_correct?'Correct':'Incorrect'}}</span></p>`;
        
        ['0', '1', '2', '3'].forEach(r => {{
            if (d.rounds[r]) {{
                const rd = d.rounds[r];
                html += `<div class="round-box">
                    <div class="round-header">Round ${{r}} (Consensus: ${{rd.consensus}})</div>`;
                    
                Object.keys(rd.responses).sort().forEach((agent, i) => {{
                    html += `<div class="agent-response">
                        <div class="agent-name">${{agent}}</div>
                        <div class="response-text">${{escapeHtml(rd.responses[agent])}}</div>
                        <div class="final-parsed">Extracted: ${{rd.final_answers[i] || 'None'}}</div>
                    </div>`;
                }});
                
                html += `</div>`;
            }}
        }});
        
        view.innerHTML = html;
    }}
    
    function escapeHtml(text) {{
        if (!text) return '';
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }}

    // Initial render
    renderList();
</script>

</body>
</html>
    """
    
    with open('debate_visualizer.html', 'w') as f:
        f.write(html_content)
    print("Visualizer generated: debate_visualizer.html")

if __name__ == "__main__":
    data = load_data()
    generate_html(data)
