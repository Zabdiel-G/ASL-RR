from flask import Flask, jsonify
import subprocess  # To run your ASL script

app = Flask(__name__)

@app.route('/api/asl', methods=['GET'])
def asl_recognition():
    # Run your original ASL recognition code and capture output
    result = subprocess.run(['python', 'asl_recognition.py'], stdout=subprocess.PIPE)
    return jsonify({'message': result.stdout.decode('utf-8')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
