from flask import Flask, render_template, request, jsonify
import json
import os
import random
import datetime

app = Flask(__name__)

# Data file path
DATA_FILE = 'lottery_data.json'

# Default test data - 20 common Chinese names
DEFAULT_NAMES = [
    "张伟", "王芳", "李娜", "刘洋", "陈明", "杨丽", "赵刚", "周婷", "吴鹏", "郑红",
    "孙亮", "马静", "朱强", "胡敏", "郭磊", "何琳", "徐杰", "林娟", "高峰", "黄丹"
]

# Initialize data structure
def init_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        data = {
            'candidates': DEFAULT_NAMES.copy(),
            'prizes': [
                {'name': '一等奖', 'count': 1, 'winners': []},
                {'name': '二等奖', 'count': 3, 'winners': []},
                {'name': '三等奖', 'count': 5, 'winners': []}
            ],
            'history': []
        }
        save_data(data)
        return data

# Save data to file
def save_data(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Helper function to get current datetime string
def get_current_datetime():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    data = init_data()
    return jsonify(data)

@app.route('/api/candidates', methods=['POST'])
def update_candidates():
    data = init_data()
    candidates = request.json.get('candidates', [])
    data['candidates'] = candidates
    save_data(data)
    return jsonify({'success': True, 'candidates': candidates})

@app.route('/api/prizes', methods=['POST'])
def update_prizes():
    data = init_data()
    prizes = request.json.get('prizes', [])
    data['prizes'] = prizes
    save_data(data)
    return jsonify({'success': True, 'prizes': prizes})

@app.route('/api/draw', methods=['POST'])
def draw_winner():
    data = init_data()
    prize_id = request.json.get('prizeId')
    
    # Find the prize
    prize = next((p for p in data['prizes'] if p['name'] == prize_id), None)
    if not prize:
        return jsonify({'success': False, 'message': 'Prize not found'})
    
    # Check if we already have enough winners
    if len(prize['winners']) >= prize['count']:
        return jsonify({'success': False, 'message': 'All winners for this prize have been drawn'})
    
    # Get available candidates (not already winners)
    all_winners = []
    for p in data['prizes']:
        all_winners.extend(p['winners'])
    
    available_candidates = [c for c in data['candidates'] if c not in all_winners]
    
    if not available_candidates:
        return jsonify({'success': False, 'message': 'No more candidates available'})
    
    # Draw a winner
    winner = random.choice(available_candidates)
    prize['winners'].append(winner)
    
    # Add to history
    data['history'].append({
        'prize': prize_id,
        'winner': winner,
        'timestamp': get_current_datetime()
    })
    
    save_data(data)
    
    return jsonify({
        'success': True,
        'winner': winner,
        'prize': prize_id,
        'remainingDraws': prize['count'] - len(prize['winners']),
        'remainingCandidates': len(available_candidates) - 1
    })

@app.route('/api/reset', methods=['POST'])
def reset_lottery():
    data = init_data()
    
    # Reset winners but keep prize settings and candidates
    for prize in data['prizes']:
        prize['winners'] = []
    
    data['history'] = []
    save_data(data)
    
    return jsonify({'success': True, 'message': 'Lottery reset successfully'})

if __name__ == '__main__':
    app.run(debug=True)
