import numpy as np
from datetime import datetime
import json
from typing import TypedDict
from langgraph.graph import StateGraph
import os
from groq import Groq
from dotenv import load_dotenv
import csv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
import resend

# -------------------------
# Load Environment Variables
# -------------------------

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

resend.api_key = os.getenv("RESEND_API_KEY")
if not resend.api_key:
    print("Warning: RESEND_API_KEY not found.")

# -------------------------
# Define Graph State
# -------------------------

class RiskState(TypedDict):
    portfolio: list
    expected_loss: float
    simulated_loss: float
    std_dev: float
    premium_pool: float
    solvency_ratio: float
    stressed_solvency_10: float
    stressed_solvency_20: float
    status: str
    narrative: str


# -------------------------
# Node 1: Generate Portfolio
# -------------------------

def generate_portfolio(state: RiskState):
    portfolio = []
    for _ in range(1000):
        policy = {
            "lambda": np.random.uniform(0.1, 0.5),
            "severity": np.random.uniform(20000, 80000),
            "premium": np.random.uniform(30000, 100000)
        }
        portfolio.append(policy)

    state["portfolio"] = portfolio
    return state


# -------------------------
# Node 2: Monte Carlo Simulation
# -------------------------

def simulate_claims(state: RiskState):

    portfolio = state["portfolio"]
    simulations = 500
    aggregate_losses = []

    expected_loss = 0
    for policy in portfolio:
        expected_loss += policy["lambda"] * policy["severity"]

    for _ in range(simulations):
        total_loss = 0
        for policy in portfolio:
            claims = np.random.poisson(policy["lambda"])
            total_loss += claims * policy["severity"]
        aggregate_losses.append(total_loss)

    state["expected_loss"] = expected_loss
    state["simulated_loss"] = aggregate_losses[-1]
    state["std_dev"] = np.std(aggregate_losses)
    state["premium_pool"] = sum(p["premium"] for p in portfolio)

    return state


# -------------------------
# Node 3: Solvency + Stress Testing
# -------------------------

def compute_solvency(state: RiskState):

    required_capital = state["expected_loss"] + 2 * state["std_dev"]
    state["solvency_ratio"] = state["premium_pool"] / required_capital

    stressed_required_10 = (state["expected_loss"] * 1.10) + 2 * state["std_dev"]
    state["stressed_solvency_10"] = state["premium_pool"] / stressed_required_10

    stressed_required_20 = (state["expected_loss"] * 1.20) + 2 * state["std_dev"]
    state["stressed_solvency_20"] = state["premium_pool"] / stressed_required_20

    return state


# -------------------------
# Node 4: Risk Classification
# -------------------------

def evaluate_risk(state: RiskState):
    sr = state["solvency_ratio"]

    if sr < 1:
        state["status"] = "High Risk - Insolvent"
    elif 1 <= sr < 1.5:
        state["status"] = "Moderate Risk - Thin Capital Buffer"
    else:
        state["status"] = "Stable - Adequate Capital"

    return state


# -------------------------
# Node 5: LLM Narrative
# -------------------------

def generate_narrative(state: RiskState):

    prompt = f"""
You are a senior actuarial risk analyst preparing a board-level summary.

Portfolio Metrics:
Expected Loss: {round(state['expected_loss'],2)}
Simulated Loss: {round(state['simulated_loss'],2)}
Standard Deviation: {round(state['std_dev'],2)}
Premium Pool: {round(state['premium_pool'],2)}
Base Solvency Ratio: {round(state['solvency_ratio'],4)}
Stressed Solvency (10% Frequency Increase): {round(state['stressed_solvency_10'],4)}
Stressed Solvency (20% Frequency Increase): {round(state['stressed_solvency_20'],4)}

Provide:
1. Capital adequacy interpretation
2. Risk exposure assessment
3. Stress scenario implications
4. Professional executive summary
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    state["narrative"] = response.choices[0].message.content
    return state


# -------------------------
# Build LangGraph Workflow
# -------------------------

builder = StateGraph(RiskState)
builder.add_node("generate_portfolio", generate_portfolio)
builder.add_node("simulate_claims", simulate_claims)
builder.add_node("compute_solvency", compute_solvency)
builder.add_node("evaluate_risk", evaluate_risk)
builder.add_node("generate_narrative", generate_narrative)

builder.set_entry_point("generate_portfolio")
builder.add_edge("generate_portfolio", "simulate_claims")
builder.add_edge("simulate_claims", "compute_solvency")
builder.add_edge("compute_solvency", "evaluate_risk")
builder.add_edge("evaluate_risk", "generate_narrative")

graph = builder.compile()


# -------------------------
# CSV Logging
# -------------------------

def append_csv(report):

    file_exists = os.path.isfile("risk_history.csv")

    with open("risk_history.csv", mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "expected_loss",
                "simulated_loss",
                "std_dev",
                "premium_pool",
                "solvency_ratio",
                "stressed_solvency_10",
                "stressed_solvency_20",
                "status"
            ])

        writer.writerow([
            report["timestamp"],
            report["expected_loss"],
            report["simulated_loss"],
            report["std_dev"],
            report["premium_pool"],
            report["solvency_ratio"],
            report["stressed_solvency_10"],
            report["stressed_solvency_20"],
            report["status"]
        ])


# -------------------------
# PDF Generator
# -------------------------

def format_currency(x):
    return f"INR {x:,.2f}"


def generate_pdf(report):

    filename = f"Risk_Report_{report['timestamp'].replace(':','-').replace(' ','_')}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>RiskLens AI</b>", styles["Title"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph("Autonomous Actuarial Risk Monitoring Report", styles["Heading2"]))
    elements.append(Spacer(1, 0.4 * inch))

    data = [
        ["Metric", "Value"],
        ["Expected Loss", format_currency(report["expected_loss"])],
        ["Simulated Loss", format_currency(report["simulated_loss"])],
        ["Standard Deviation", format_currency(report["std_dev"])],
        ["Premium Pool", format_currency(report["premium_pool"])],
        ["Base Solvency Ratio", f"{report['solvency_ratio']:.4f}"],
        ["Stress Solvency (10%)", f"{report['stressed_solvency_10']:.4f}"],
        ["Stress Solvency (20%)", f"{report['stressed_solvency_20']:.4f}"],
        ["Capital Status", report["status"]],
    ]

    table = Table(data, colWidths=[230, 230])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E4053")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.5 * inch))

    clean_text = report["narrative"].replace("**", "")
    for para in clean_text.split("\n\n"):
        elements.append(Paragraph(para.strip(), styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(
        f"Confidential – Internal Risk Monitoring | Generated on: {report['timestamp']}",
        styles["Normal"]
    ))

    doc.build(elements)


# -------------------------
# Email Alert
# -------------------------

def send_breach_email(report):
    try:
        resend.Emails.send({
            "from": "RiskLens <onboarding@resend.dev>",
            "to": [os.getenv("ALERT_EMAIL")],
            "subject": "⚠ RiskLens AI Alert: Capital Threshold Breach",
            "html": f"""
            <h2>Capital Adequacy Alert</h2>
            <p><strong>Base Solvency Ratio:</strong> {report['solvency_ratio']:.4f}</p>
            <p><strong>Stress 20% Solvency:</strong> {report['stressed_solvency_20']:.4f}</p>
            <p>Please review immediately.</p>
            """
        })
        print("Breach alert email sent.")
    except Exception as e:
        print("Email sending failed:", e)


# -------------------------
# Main Execution
# -------------------------

def run_agent():

    final_state = graph.invoke({})

    report = {
        "timestamp": str(datetime.now()),
        "expected_loss": float(final_state["expected_loss"]),
        "simulated_loss": float(final_state["simulated_loss"]),
        "std_dev": float(final_state["std_dev"]),
        "premium_pool": float(final_state["premium_pool"]),
        "solvency_ratio": float(final_state["solvency_ratio"]),
        "stressed_solvency_10": float(final_state["stressed_solvency_10"]),
        "stressed_solvency_20": float(final_state["stressed_solvency_20"]),
        "status": final_state["status"],
        "narrative": final_state["narrative"]
    }

    # Breach Logic (TEST MODE)
    if report["solvency_ratio"] < 4:
        print("Triggering test email...")
        send_breach_email(report)

    # Logging
    with open("risk_agent_log.json", "a") as f:
        f.write(json.dumps(report) + "\n")

    append_csv(report)
    generate_pdf(report)

    print("Report generated successfully.")
    print(report)
    return report

if __name__ == "__main__":
    run_agent()