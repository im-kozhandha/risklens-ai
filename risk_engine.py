import numpy as np
from datetime import datetime
import json

def generate_portfolio(n_policies=1000):
    portfolio = []
    for _ in range(n_policies):
        policy = {
            "lambda": np.random.uniform(0.1, 0.5),  # claim frequency
            "severity": np.random.uniform(20000, 80000),
            "premium": np.random.uniform(30000, 100000)
        }
        portfolio.append(policy)
    return portfolio


def simulate_claims(portfolio):
    total_loss = 0
    expected_loss = 0
    losses = []

    for policy in portfolio:
        lam = policy["lambda"]
        sev = policy["severity"]

        claims = np.random.poisson(lam)
        loss = claims * sev

        total_loss += loss
        expected_loss += lam * sev
        losses.append(loss)

    std_dev = np.std(losses)
    return total_loss, expected_loss, std_dev


def compute_solvency(expected_loss, std_dev, premium_pool):
    required_capital = expected_loss + 2 * std_dev
    solvency_ratio = premium_pool / required_capital
    return solvency_ratio


def run_risk_cycle():
    portfolio = generate_portfolio()
    total_loss, expected_loss, std_dev = simulate_claims(portfolio)

    premium_pool = sum(p["premium"] for p in portfolio)
    solvency_ratio = compute_solvency(expected_loss, std_dev, premium_pool)

    result = {
        "timestamp": str(datetime.now()),
        "expected_loss": expected_loss,
        "simulated_loss": total_loss,
        "std_dev": std_dev,
        "premium_pool": premium_pool,
        "solvency_ratio": solvency_ratio
    }

    with open("risk_log.json", "a") as f:
        f.write(json.dumps(result) + "\n")

    print(result)


if __name__ == "__main__":
    run_risk_cycle()