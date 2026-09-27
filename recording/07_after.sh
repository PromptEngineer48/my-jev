source /workspace/my-jev/recording/demo.sh
say "AFTER training: the same tricky question that the base model got wrong."
run ".venv-serve/bin/python ask.py examples/charge-dispute.json"
say "Now let's ask OUR model and TypeSafe Jev the same questions, side by side."
run ".venv-serve/bin/python ask.py examples/charge-dispute.json --jev"
run ".venv-serve/bin/python ask.py examples/sentiment.json --jev"
run ".venv-serve/bin/python ask.py examples/yes-no.json --jev"
