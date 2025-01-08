from flask import Flask, request, jsonify
import os
import socket

app = Flask(__name__)

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def catch_all(path):
    headers = dict(request.headers)
    try:
        body = request.get_data(as_text=True)
    except Exception as e:
        body = str(e)

    # More robust way to get hostname:
    # 1. Try to get from environment variable first (common in container environments)
    # 2. Try to get from socket.gethostname() as fallback
    # 3. Use 'Unknown Hostname' as final fallback
    hostname = os.environ.get('HOSTNAME') or socket.gethostname() or 'Unknown Hostname'

    response_data = {
        "path": path,
        "method": request.method,
        "headers": headers,
        "body": body,
        "hostname": hostname
    }

    return jsonify(response_data)

port=int(os.getenv("FLASK_APP_PORT", 5000))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=port)