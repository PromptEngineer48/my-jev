# My Jev vs TypeSafe Jev

| Metric | My Jev (Unsloth 4B) | TypeSafe Jev |
|---|---:|---:|
| Accuracy: phishing | 47.7% | 63.7% |
| Accuracy: tev1_dev | 85.8% | 80.0% |
| Accuracy: ticket_routing | 88.9% | 100.0% |
| Accuracy: tool_risk | 88.3% | 91.7% |
| **Accuracy: overall** | **70.8%** | **75.1%** |
| Brier (lower better) | 0.435 | 0.354 |
| ECE (lower better) | 0.212 | 0.105 |
| Latency p50 (ms) | 461 | 357 |
| Latency p90 (ms) | 582 | 393 |
| Latency p99 (ms) | 776 | 515 |
| Latency mean (ms) | 471 | 365 |
| Burst throughput (req/s) | 22.3 | 38.92 |
| Cost per 1k decisions | $0.0061 | $0.0232 |
| Errors | 0 | 0 |
