import json
import glob
import os
import re

HISTORY_DIR = "/export/home3/dazhou/debate-or-vote/out/history"

def load_data():
    # Search all subdirs but exclude backup directories
    all_files = glob.glob(os.path.join(HISTORY_DIR, "**", "*.jsonl"), recursive=True)
    files = [f for f in all_files if "previous_backup" not in f]
    
    all_data = []
    
    # Updated regex to capture dataset, model, TR, TT
    pattern = re.compile(r"([a-zA-Z0-9_]+)_\d+__(.+?)_N=\d+_R=\d+_TR=(\d+)_TT=([\d\.]+)\.jsonl")
    
    for f in files:
        basename = os.path.basename(f)
        match = pattern.match(basename)
        if not match:
            continue
        
        dataset_name = match.group(1)
        model_name = match.group(2)
        tr = int(match.group(3))
        tt = float(match.group(4))
        
        with open(f, 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                if not line.strip():
                    continue
                record = json.loads(line)
                
                answer = record.get('0', {}).get('answer', 'N/A')
                
                # Process rounds
                rounds_info = {}
                keys = sorted([k for k in record.keys() if k.isdigit()], key=int)
                
                for r_key in keys:
                    r_data = record[r_key]
                    responses = r_data.get('responses', {})
                    agents_resps = {}
                    for k, v in responses.items():
                        short_k = k.split("__")[-1] 
                        agents_resps[short_k] = v
                    
                    final_answers = r_data.get('final_answers', [])
                    
                    rounds_info[r_key] = {
                        "responses": agents_resps,
                        "final_answers": final_answers,
                        "consensus": r_data.get('debate_answer'),
                        "is_correct": r_data.get('debate_answer_iscorr', False)
                    }

                last_round_key = keys[-1] if keys else None
                final_correct = False
                if last_round_key:
                    final_correct = rounds_info[last_round_key]["is_correct"]

                entry = {
                    "id": idx,
                    "file": basename,
                    "dataset": dataset_name,
                    "model": model_name,
                    "TR": tr,
                    "TT": tt,
                    "is_correct": final_correct,
                    "answer": answer,
                    "rounds": rounds_info,
                    "round_keys": keys
                }
                all_data.append(entry)
    
    print(f"Loaded {len(all_data)} records from {len(files)} files")
    datasets = sorted(set(e['dataset'] for e in all_data))
    models = sorted(set(e['model'] for e in all_data))
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    return all_data

def generate_html(data):
    # Sort data: dataset -> model -> TR -> TT -> ID
    data.sort(key=lambda x: (x['dataset'], x['model'], x['TR'], x['TT'], x['id']))
    
    data_json = json.dumps(data)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Debate History Visualizer (Multi-Model)</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #eaeff2; color: #333; }}
        
        .container {{ display: flex; gap: 20px; height: 85vh; }}
        .controls {{ margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
        .controls label {{ font-weight: bold; font-size: 0.9em; color: #555; }}
        .controls select {{ padding: 6px 10px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.9em; min-width: 120px; }}
        
        .stats-bar {{ margin-bottom: 10px; padding: 10px 15px; background: #fff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); display: flex; gap: 20px; font-size: 0.85em; color: #666; }}
        .stats-bar .stat {{ display: flex; align-items: center; gap: 5px; }}
        .stats-bar .stat-value {{ font-weight: bold; color: #333; }}
        
        .sidebar {{ width: 320px; min-width: 320px; background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); display: flex; flex-direction: column; }}
        .sidebar-header {{ padding: 15px; border-bottom: 1px solid #eee; font-weight: bold; background: #f8f9fa; border-radius: 8px 8px 0 0; }}
        .list-container {{ flex: 1; overflow-y: auto; padding: 0; }}
        
        .item-group {{ border-bottom: 1px solid #eee; }}
        .item-header {{ padding: 10px 15px; cursor: pointer; display: flex; align-items: center; justify-content: space-between; background: white; }}
        .item-header:hover {{ background: #f1f3f5; }}
        .item-header.active {{ background: #e7f5ff; border-left: 4px solid #007bff; }}
        
        .item-info {{ font-size: 0.9em; }}
        .item-id {{ font-weight: bold; color: #444; }}
        .item-meta {{ color: #888; font-size: 0.85em; }}
        
        .round-list {{ display: none; background: #fafafa; border-top: 1px solid #eee; }}
        .round-list.open {{ display: block; }}
        .round-item {{ padding: 6px 15px 6px 35px; cursor: pointer; font-size: 0.85em; color: #666; display: flex; align-items: center; }}
        .round-item:hover {{ background: #eee; }}
        .round-item.selected {{ background: #d0ebff; color: #004085; font-weight: bold; }}

        .dot {{ height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 8px; flex-shrink: 0; }}
        .correct {{ background-color: #28a745; }}
        .incorrect {{ background-color: #dc3545; }}
        
        .content {{ flex: 1; background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); padding: 25px; overflow-y: auto; }}
        
        .detail-meta {{ background: #f8f9fa; padding: 12px 15px; border: 1px solid #eee; border-radius: 6px; display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; font-size: 0.9em; margin-bottom: 20px; }}
        .detail-meta .meta-item {{ display: flex; flex-direction: column; }}
        .detail-meta .meta-label {{ font-size: 0.8em; color: #888; text-transform: uppercase; }}
        .detail-meta .meta-value {{ font-weight: bold; color: #333; }}
        
        .round-section {{ border: 1px solid #e9ecef; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); overflow: hidden; opacity: 0.6; transition: opacity 0.2s; }}
        .round-section.focused {{ opacity: 1.0; border-color: #007bff; box-shadow: 0 0 0 2px rgba(0,123,255,0.25); }}
        
        .r-header {{ background: #f8f9fa; padding: 10px 15px; border-bottom: 1px solid #e9ecef; display: flex; justify-content: space-between; align-items: center; }}
        .r-title {{ font-weight: bold; font-size: 1.1em; }}
        
        .agents-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; padding: 15px; }}
        .agent-card {{ background: #fff; border: 1px solid #eee; border-radius: 6px; padding: 10px; }}
        .agent-name {{ font-weight: bold; color: #495057; border-bottom: 2px solid #e9ecef; padding-bottom: 5px; margin-bottom: 10px; font-size: 0.9em; }}
        .agent-text {{ font-family: Consolas, Monaco, 'Courier New', monospace; font-size: 0.85em; white-space: pre-wrap; color: #333; max-height: 250px; overflow-y: auto; background: #fcfcfc; padding: 8px; border-radius: 4px; }}
        
        .parsed-ans {{ margin-top: 8px; font-size: 0.85em; color: #6610f2; font-weight: bold; }}
        
        .highlight-bar {{ height: 5px; background: #007bff; width: 0%; transition: width 0.3s; }}
        .round-section.focused .highlight-bar {{ width: 100%; }}
        
        .placeholder {{ text-align: center; margin-top: 80px; color: #aaa; font-size: 1.1em; }}
    </style>
</head>
<body>

<div class="controls">
    <label>Dataset:</label>
    <select id="datasetFilter" onchange="onDatasetChange()">
        <option value="all">All</option>
    </select>
    
    <label>Model:</label>
    <select id="modelFilter" onchange="onModelChange()">
        <option value="all">All</option>
    </select>
    
    <label>Target Round:</label>
    <select id="trFilter" onchange="filterAndRender()">
        <option value="all">All</option>
    </select>
    
    <label>Temperature:</label>
    <select id="ttFilter" onchange="filterAndRender()">
        <option value="all">All</option>
    </select>
    
    <label>Status:</label>
    <select id="statusFilter" onchange="filterAndRender()">
        <option value="all">All</option>
        <option value="correct">Final Correct</option>
        <option value="incorrect">Final Incorrect</option>
    </select>
</div>

<div class="stats-bar" id="statsBar">
    <div class="stat">Items: <span class="stat-value" id="statCount">0</span></div>
    <div class="stat">Correct: <span class="stat-value" id="statCorrect" style="color:#28a745;">0</span></div>
    <div class="stat">Incorrect: <span class="stat-value" id="statIncorrect" style="color:#dc3545;">0</span></div>
    <div class="stat">Accuracy: <span class="stat-value" id="statAccuracy">0%</span></div>
</div>

<div class="container">
    <div class="sidebar">
        <div class="sidebar-header" id="countDisplay">Items</div>
        <div class="list-container" id="itemList"></div>
    </div>
    
    <div class="content" id="detailView">
        <div class="placeholder">Select an item from the sidebar to view details</div>
    </div>
</div>

<script>
    const allData = {data_json};
    let currentData = allData;
    
    // Helper: populate a <select> with values, keeping the first "All" option
    function populateSelect(selectId, values, labelFn) {{
        const select = document.getElementById(selectId);
        while (select.options.length > 1) select.remove(1);
        values.forEach(v => {{
            let opt = document.createElement('option');
            opt.value = v;
            opt.innerText = labelFn ? labelFn(v) : v;
            select.appendChild(opt);
        }});
    }}
    
    // Helper: get filtered subset based on specific filters only
    function getSubset(filters) {{
        return allData.filter(d => {{
            if (filters.dataset && filters.dataset !== 'all' && d.dataset !== filters.dataset) return false;
            if (filters.model && filters.model !== 'all' && d.model !== filters.model) return false;
            if (filters.tr && filters.tr !== 'all' && d.TR != filters.tr) return false;
            if (filters.tt && filters.tt !== 'all' && d.TT != filters.tt) return false;
            return true;
        }});
    }}
    
    // Init all filter dropdowns
    function initFilters() {{
        populateSelect('datasetFilter', [...new Set(allData.map(d => d.dataset))].sort());
        populateSelect('modelFilter', [...new Set(allData.map(d => d.model))].sort());
        populateSelect('trFilter', [...new Set(allData.map(d => d.TR))].sort((a,b)=>a-b), v => 'TR = '+v);
        populateSelect('ttFilter', [...new Set(allData.map(d => d.TT))].sort((a,b)=>a-b), v => 'T = '+v);
    }}
    
    // When dataset changes, update model/TR/TT options
    function onDatasetChange() {{
        const dsVal = document.getElementById('datasetFilter').value;
        const subset = getSubset({{dataset: dsVal}});
        
        populateSelect('modelFilter', [...new Set(subset.map(d => d.model))].sort());
        populateSelect('trFilter', [...new Set(subset.map(d => d.TR))].sort((a,b)=>a-b), v => 'TR = '+v);
        populateSelect('ttFilter', [...new Set(subset.map(d => d.TT))].sort((a,b)=>a-b), v => 'T = '+v);
        
        filterAndRender();
    }}
    
    // When model changes, update TR/TT options
    function onModelChange() {{
        const dsVal = document.getElementById('datasetFilter').value;
        const modelVal = document.getElementById('modelFilter').value;
        const subset = getSubset({{dataset: dsVal, model: modelVal}});
        
        populateSelect('trFilter', [...new Set(subset.map(d => d.TR))].sort((a,b)=>a-b), v => 'TR = '+v);
        populateSelect('ttFilter', [...new Set(subset.map(d => d.TT))].sort((a,b)=>a-b), v => 'T = '+v);
        
        filterAndRender();
    }}

    function filterAndRender() {{
        const dsVal = document.getElementById('datasetFilter').value;
        const modelVal = document.getElementById('modelFilter').value;
        const trVal = document.getElementById('trFilter').value;
        const ttVal = document.getElementById('ttFilter').value;
        const statusVal = document.getElementById('statusFilter').value;
        
        currentData = allData.filter(d => {{
            if (dsVal !== 'all' && d.dataset !== dsVal) return false;
            if (modelVal !== 'all' && d.model !== modelVal) return false;
            if (trVal !== 'all' && d.TR != trVal) return false;
            if (ttVal !== 'all' && d.TT != ttVal) return false;
            if (statusVal === 'correct' && !d.is_correct) return false;
            if (statusVal === 'incorrect' && d.is_correct) return false;
            return true;
        }});
        
        const total = currentData.length;
        const correctCount = currentData.filter(d => d.is_correct).length;
        const incorrectCount = total - correctCount;
        const accuracy = total > 0 ? (correctCount / total * 100).toFixed(1) : 0;
        
        document.getElementById('statCount').innerText = total;
        document.getElementById('statCorrect').innerText = correctCount;
        document.getElementById('statIncorrect').innerText = incorrectCount;
        document.getElementById('statAccuracy').innerText = accuracy + '%';
        document.getElementById('countDisplay').innerText = 'Showing ' + total + ' Items';
        
        renderSidebar();
    }}
    
    function renderSidebar() {{
        const list = document.getElementById('itemList');
        list.innerHTML = '';
        
        currentData.forEach(d => {{
            const group = document.createElement('div');
            group.className = 'item-group';
            
            const header = document.createElement('div');
            header.className = 'item-header';
            header.innerHTML = 
                '<div class="item-info">' +
                    '<div class="item-id">ID ' + d.id + ' — ' + d.dataset + '</div>' +
                    '<div class="item-meta">' + d.model + ' | TR=' + d.TR + ' T=' + d.TT + '</div>' +
                '</div>' +
                '<div class="dot ' + (d.is_correct ? 'correct' : 'incorrect') + '" title="Final Result"></div>';
            
            const roundList = document.createElement('div');
            roundList.className = 'round-list';
            
            d.round_keys.forEach(r => {{
                const rInfo = d.rounds[r];
                const rItem = document.createElement('div');
                rItem.className = 'round-item';
                rItem.innerHTML = '<div class="dot ' + (rInfo.is_correct ? 'correct' : 'incorrect') + '"></div>Round ' + r;
                rItem.onclick = function(e) {{
                    e.stopPropagation();
                    showDetail(d, r);
                    document.querySelectorAll('.round-item').forEach(function(el) {{ el.classList.remove('selected'); }});
                    rItem.classList.add('selected');
                }};
                roundList.appendChild(rItem);
            }});
            
            header.onclick = function() {{
                const isOpen = roundList.classList.contains('open');
                document.querySelectorAll('.round-list').forEach(function(el) {{ el.classList.remove('open'); }});
                document.querySelectorAll('.item-header').forEach(function(el) {{ el.classList.remove('active'); }});
                
                if (!isOpen) {{
                    roundList.classList.add('open');
                    header.classList.add('active');
                    showDetail(d, d.round_keys[d.round_keys.length-1]);
                }}
            }};
            
            group.appendChild(header);
            group.appendChild(roundList);
            list.appendChild(group);
        }});
    }}
    
    function showDetail(d, focusRound) {{
        const view = document.getElementById('detailView');
        const rDetails = d.rounds[focusRound];
        
        let html = '<h2 style="margin-top:0;">Problem ID ' + d.id + '</h2>';
        html += '<div class="detail-meta">';
        html += '<div class="meta-item"><span class="meta-label">Dataset</span><span class="meta-value">' + d.dataset + '</span></div>';
        html += '<div class="meta-item"><span class="meta-label">Model</span><span class="meta-value">' + d.model + '</span></div>';
        html += '<div class="meta-item"><span class="meta-label">Target Round</span><span class="meta-value">' + d.TR + '</span></div>';
        html += '<div class="meta-item"><span class="meta-label">Temperature</span><span class="meta-value">' + d.TT + '</span></div>';
        html += '<div class="meta-item"><span class="meta-label">Correct Answer</span><span class="meta-value">' + d.answer + '</span></div>';
        html += '<div class="meta-item"><span class="meta-label">Focus Round ' + focusRound + '</span><span class="meta-value" style="color:' + (rDetails.is_correct ? '#28a745' : '#dc3545') + '">' + (rDetails.is_correct ? 'Correct' : 'Incorrect') + '</span></div>';
        html += '</div>';
        
        d.round_keys.forEach(function(r) {{
            const isFocused = (r == focusRound);
            const info = d.rounds[r];
            
            html += '<div id="round-box-' + r + '" class="round-section ' + (isFocused ? 'focused' : '') + '">';
            html += '<div class="highlight-bar"></div>';
            html += '<div class="r-header"><div class="r-title">Round ' + r + '</div>';
            html += '<div>Consensus: <strong>' + info.consensus + '</strong> <span class="dot ' + (info.is_correct ? 'correct' : 'incorrect') + '" style="margin-left:5px;"></span></div></div>';
            html += '<div class="agents-grid">';
            
            const agentKeys = Object.keys(info.responses).sort();
            agentKeys.forEach(function(agent, i) {{
                html += '<div class="agent-card">';
                html += '<div class="agent-name">' + agent + '</div>';
                html += '<div class="agent-text">' + escapeHtml(info.responses[agent]) + '</div>';
                html += '<div class="parsed-ans">Extracted: ' + (info.final_answers[i] !== undefined ? info.final_answers[i] : 'N/A') + '</div>';
                html += '</div>';
            }});
            
            html += '</div></div>';
        }});
        
        view.innerHTML = html;
        
        setTimeout(function() {{
            const el = document.getElementById('round-box-' + focusRound);
            if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}, 100);
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
    
    // Initial setup
    initFilters();
    filterAndRender();
</script>

</body>
</html>
    """
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'debate_visualizer_v2.html')
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"Visualizer V2 generated: {output_path}")

if __name__ == "__main__":
    data = load_data()
    generate_html(data)
