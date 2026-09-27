source /workspace/my-jev/recording/demo.sh
say "BEFORE training: let's ask the plain Qwen3.5-4B model, served with vLLM."
say "Question 1: a customer was charged twice. Correct answer = duplicate_charge."
run "cat examples/charge-dispute.json"
run ".venv-serve/bin/python ask.py examples/charge-dispute.json --label 'Base Qwen3.5-4B (untrained)'"
say "Wrong! The base model thinks the customer wants to cancel. Let's try an easy yes/no one."
run ".venv-serve/bin/python ask.py examples/return-window.json --label 'Base Qwen3.5-4B (untrained)'"
say "Easy questions are fine. Tricky intent questions are not. That's what fine-tuning fixes."
