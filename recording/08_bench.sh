source /workspace/my-jev/recording/demo.sh
say "Step 6: the big test. 747 questions, our model vs TypeSafe Jev."
say "Suites: held-out test set, phishing emails, tool risk, ticket routing."
run ".venv-serve/bin/python bench.py --gpu_hourly 0.49"
