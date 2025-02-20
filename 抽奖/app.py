from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

def get_names_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f]
        return names
    except FileNotFoundError:
        return None

@app.route('/')
def index():
    names = get_names_from_file('names.txt')
    if names:
        return render_template('index.html', names=names)
    else:
        return render_template('index.html', error='名单文件未找到，请确保 names.txt 文件存在')

@app.route('/draw', methods=['POST'])
def draw():
    num_winners = int(request.form.get('num_winners', 1))
    names = get_names_from_file('names.txt')
    if names:
        if num_winners > len(names):
            return jsonify({'error': '中奖人数不能超过名单总人数'})
        winners = random.sample(names, num_winners)
        return jsonify({'names': names, 'winners': winners}) # Return all names and winners
    else:
        return jsonify({'error': '名单文件未找到，请先上传'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
