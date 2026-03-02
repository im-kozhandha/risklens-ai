from flask import Flask, jsonify
from risk_agent import run_agent

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "RiskLens AI is running."}

@app.route("/run")
def run():
    result = run_agent()
    return jsonify(result)

@app.route("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)