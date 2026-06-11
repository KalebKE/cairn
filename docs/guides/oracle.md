# Oracle Intelligence

## Overview

The OMEGA Oracle tracks predictions, measures calibration accuracy via Brier scoring, detects regime changes, and generates strategy playbooks from your forecast history. It is a compounding intelligence layer: the more predictions you record and resolve, the more precise your calibration analysis becomes.

The Oracle works with any prediction domain (crypto markets, project outcomes, hiring decisions) but was designed with probabilistic markets in mind.

## Quick Example

```
# Record a prediction
omega_oracle_record(
    record_type="prediction",
    content="The beta launch will reach 100 signups by end of March",
    data={"market_id": "beta-100-signups-march", "our_probability": 0.72, "market_price": 0.65, "edge": 0.07}
)

# Later, resolve it
omega_oracle_resolve(market_id="beta-100-signups-march", outcome="yes")

# Analyze your calibration
omega_oracle_analyze(view="calibration")

# Get a full briefing
omega_oracle_analyze(view="briefing")
```

## Tools Reference

| Tool | Purpose |
|------|---------|
| `omega_oracle_record` | Record a prediction, wallet score, regime change, or signal snapshot. Accepts structured `data` with domain-specific fields. |
| `omega_oracle_resolve` | Mark a prediction as resolved (yes/no outcome) by `market_id`. Computes P&L. |
| `omega_oracle_analyze` | Compute analytical views: `calibration`, `signals`, `wallets`, `bias`, `playbook`, or `briefing` (composite). |
| `omega_oracle_status` | Dashboard: prediction count, resolved count, Brier score, active regime, signal/wallet coverage. |

## Record Types

| Type | Purpose | Key Data Fields |
|------|---------|-----------------|
| `prediction` | A probabilistic forecast | `market_id`, `our_probability`, `market_price`, `edge`, `confidence`, `signals_present`, `regime`, `time_horizon_days` |
| `wallet_score` | Track a wallet's accuracy | `address`, `brier_score`, `trade_count`, `win_rate`, `is_informed` |
| `regime_change` | Log a market regime transition | `from_regime`, `to_regime`, `confidence` |
| `signal_snapshot` | Capture raw signal data for later analysis | Signal-specific fields |

## Analytical Views

| View | What It Computes |
|------|-----------------|
| `calibration` | Brier score, calibration curve (predicted vs actual probability), overconfidence/underconfidence detection |
| `signals` | Per-signal accuracy: which signals were present in correct vs incorrect predictions |
| `wallets` | Top wallets ranked by prediction accuracy. Identifies informed traders. |
| `bias` | Systematic mispricings: where your predictions consistently deviate from outcomes |
| `playbook` | Optimal strategy per regime based on historical performance |
| `briefing` | Composite of all views. Use as pre-session context for probability estimation tasks. |

## Common Workflows

### Track a Prediction

Record the prediction with structured data:

```
omega_oracle_record(
    record_type="prediction",
    content="ETH will flip 4k before Q2",
    data={
        "market_id": "eth-4k-q2",
        "our_probability": 0.45,
        "market_price": 0.38,
        "edge": 0.07,
        "confidence": "medium",
        "signals_present": ["momentum", "on-chain-accumulation"],
        "regime": "bull",
        "time_horizon_days": 90
    }
)
```

### Resolve When Outcome is Known

```
omega_oracle_resolve(market_id="eth-4k-q2", outcome="no")
```

### Check Overall Calibration

```
omega_oracle_analyze(view="calibration", days=90)
# Returns: Brier score, calibration curve, sample size
```

A Brier score of 0.0 is perfect; 0.25 is random. Aim for below 0.15 for useful predictions.

### Get Pre-Session Briefing

Before a prediction session, load historical context:

```
omega_oracle_analyze(view="briefing", market_type="btc")
# Returns: calibration, recent signals, wallet rankings, bias patterns, regime playbook
```

### Track Regime Changes

```
omega_oracle_record(
    record_type="regime_change",
    content="Market shifted from accumulation to markup phase",
    data={"from_regime": "accumulation", "to_regime": "markup", "confidence": 0.85}
)
```

### Dashboard

Quick status check:

```
omega_oracle_status()
# Returns: 47 predictions, 31 resolved, Brier: 0.12, regime: markup, 5 signals tracked
```

## Tips

- **Resolve predictions promptly.** Unresolved predictions cannot contribute to calibration analysis. The more resolved data, the more accurate your Brier score.
- **Use the briefing view before prediction sessions.** It aggregates all analytical views into a single context dump, giving you historical awareness before making new forecasts.
- **Track signals consistently.** The `signals_present` field on predictions enables per-signal accuracy analysis. If you always record which signals were present, you can identify which ones actually predict outcomes.
- **Brier score improves with volume.** Below 30 resolved predictions, the score is noisy. Above 100, it becomes a reliable measure of your forecasting ability.
- **Regime awareness matters.** Strategies that work in one regime (e.g., bull market) may fail in another. Use the `playbook` view to see regime-specific performance.
- **Market type scoping.** Use `market_type` to separate predictions by domain (e.g., "btc", "eth", "hiring"). This keeps calibration analysis clean across different prediction domains.
