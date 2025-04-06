from flask import Flask, render_template, request, jsonify
import random
import json
import os

app = Flask(__name__)

# Initialize data
lottery_data = {
    "prizes": [],
    "candidates": [],
    "winners": []
}

# Load test data
def load_test_data():
    test_names = [
        "张伟", "王芳", "李娜", "刘洋", "陈明", 
        "杨丽", "赵刚", "周婷", "吴鹏", "郑红",
        "孙亮", "马静", "朱强", "胡敏", "郭磊",
        "何琳", "徐杰", "林娟", "高峰", "黄丹"
    ]
    lottery_data["candidates"] = test_names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_lottery', methods=['POST'])
def start_lottery():
    data = request.json
    prize_name = data.get('prize_name')
    winner_count = data.get('winner_count', 1)
    
    if len(lottery_data["candidates"]) < winner_count:
        return jsonify({"error": "Not enough candidates"}), 400
    
    winners = random.sample(lottery_data["candidates"], winner_count)
    lottery_data["winners"].extend([{"name": w, "prize": prize_name} for w in winners])
    
    # Remove winners from candidates
    lottery_data["candidates"] = [c for c in lottery_data["candidates"] if c not in winners]
    
    return jsonify({
        "winners": winners,
        "remaining_candidates": len(lottery_data["candidates"])
    })

@app.route('/get_candidates', methods=['GET'])
def get_candidates():
    return jsonify({
        "candidates": lottery_data["candidates"],
        "count": len(lottery_data["candidates"])
    })

if __name__ == '__main__':
    load_test_data()
    app.run(debug=True)
