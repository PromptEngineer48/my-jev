# My Jev vs TypeSafe Jev

| Metric | My Jev (Unsloth 4B) | TypeSafe Jev |
|---|---:|---:|
| Accuracy: phishing | 47.7% | 64.0% |
| Accuracy: tev1_dev | 85.8% | 80.6% |
| Accuracy: ticket_routing | 88.9% | 100.0% |
| Accuracy: tool_risk | 88.3% | 91.7% |
| **Accuracy: overall** | **70.8%** | **75.5%** |
| Brier (lower better) | 0.435 | 0.355 |
| ECE (lower better) | 0.212 | 0.111 |
| Latency p50 (ms) | 118 | 269 |
| Latency p90 (ms) | 192 | 301 |
| Latency p99 (ms) | 283 | 411 |
| Latency mean (ms) | 132 | 273 |
| Burst throughput (req/s) | 32.91 | 51.58 |
| Cost per 1k decisions | $0.0041 | $0.0232 |
| Errors | 0 | 0 |
