from flask import Flask, render_template, request, jsonify, session
import random
import json
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production

# Initialize session variables
@app.before_request
def before_request():
    if 'winners' not in session:
        session['winners'] = []
    if 'history' not in session:
        session['history'] = []
    if 'remaining_names' not in session:
        session['remaining_names'] = get_names_from_file('names.txt') or []

def get_names_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        return names
    except FileNotFoundError:
        return None

def save_history(prize_name, winners):
    session['history'].append({
        'prize': prize_name,
        'winners': winners,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    session.modified = True

@app.route('/')
def index():
    names = get_names_from_file('names.txt')
    if names:
        return render_template('index.html', 
                            names=names,
                            remaining=len(session['remaining_names']),
                            history=session['history'])
    else:
        return render_template('index.html', 
                            error='名单文件未找到，请确保 names.txt 文件存在',
                            remaining=0,
                            history=session['history'])

@app.route('/draw', methods=['POST'])
def draw():
    prize_name = request.form.get('prize_name', '幸运奖')
    num_winners = int(request.form.get('num_winners', 1))
    
    # Ensure we're only selecting from remaining names that haven't won yet
    available_names = [name for name in session['remaining_names'] if name not in session['winners']]
    if not available_names:
        return jsonify({'error': '没有可用的候选人'})
    
    if num_winners > len(available_names):
        return jsonify({'error': f'剩余候选人不足，只有 {len(available_names)} 人'})
    
    winners = random.sample(available_names, num_winners)
    session['remaining_names'] = [name for name in session['remaining_names'] if name not in winners]
    session['winners'].extend(winners)
    save_history(prize_name, winners)
    
    return jsonify({
        'all_names': session['remaining_names'] + session['winners'],
        'remaining': len(session['remaining_names']),
        'current_winners': winners,
        'history': session['history']
    })

@app.route('/reset', methods=['POST'])
def reset():
    session['winners'] = []
    session['remaining_names'] = get_names_from_file('names.txt') or []
    session['history'] = []
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)