from flask import Flask, render_template, request, jsonify
import random
import json
import os

app = Flask(__name__)

# 数据存储
data_file = "lottery_data.json"


# 初始化数据
def init_data():
    default_candidates = [
        "张伟",
        "王芳",
        "李娜",
        "刘洋",
        "陈明",
        "杨丽",
        "赵刚",
        "周婷",
        "吴鹏",
        "郑红",
        "孙亮",
        "马静",
        "朱强",
        "胡敏",
        "郭磊",
        "何琳",
        "徐杰",
        "林娟",
        "高峰",
        "黄丹",
    ]

    default_data = {"candidates": default_candidates, "winners": [], "history": []}

    if not os.path.exists(data_file):
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=4)

    return default_data


# 获取当前数据
def get_data():
    if not os.path.exists(data_file):
        return init_data()

    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


# 保存数据
def save_data(data):
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 路由
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/candidates", methods=["GET"])
def get_candidates():
    data = get_data()
    return jsonify(
        {
            "candidates": data["candidates"],
            "winners": data["winners"],
            "remaining": [c for c in data["candidates"] if c not in data["winners"]],
        }
    )


@app.route("/api/candidates", methods=["POST"])
def update_candidates():
    data = get_data()
    new_candidates = request.json.get("candidates", [])
    data["candidates"] = new_candidates
    data["winners"] = []
    data["history"] = []
    save_data(data)
    return jsonify({"success": True})


@app.route("/api/draw", methods=["POST"])
def draw_winners():
    data = get_data()

    prize_name = request.json.get("prizeName", "奖项")
    count = int(request.json.get("count", 1))

    # 获取未中奖的候选人
    remaining = [c for c in data["candidates"] if c not in data["winners"]]

    # 确保抽奖人数不超过剩余候选人数
    if count > len(remaining):
        count = len(remaining)

    # 随机抽取获奖者
    new_winners = random.sample(remaining, count) if remaining else []

    # 更新获奖者列表
    data["winners"].extend(new_winners)

    # 记录历史
    data["history"].append(
        {"prize": prize_name, "winners": new_winners, "timestamp": import_time()}
    )

    save_data(data)

    return jsonify(
        {
            "prize": prize_name,
            "winners": new_winners,
            "remaining": len(remaining) - len(new_winners),
        }
    )


@app.route("/api/reset", methods=["POST"])
def reset_lottery():
    data = get_data()
    data["winners"] = []
    save_data(data)
    return jsonify({"success": True})


@app.route("/api/history", methods=["GET"])
def get_history():
    data = get_data()
    return jsonify({"history": data["history"]})


def import_time():
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    init_data()
    app.run(debug=True, port=5001)
