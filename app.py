from flask import Flask, jsonify
from risk_agent import run_agent
from flask import render_template
import csv

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "RiskLens AI is running."}

@app.route("/dashboard")
def dashboard():
    data = []

    try:
        with open("risk_history.csv", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        data = []

    latest = data[-1] if data else None

    return render_template("dashboard.html", latest=latest, history=data)

@app.route("/run")
def run():
    result = run_agent()
    return jsonify(result)

@app.route("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)