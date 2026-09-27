source /workspace/my-jev/recording/demo.sh
say "Step 2: one script installs Unsloth, installs vLLM, and builds the training data."
run "cat setup_pod.sh"
run "bash setup_pod.sh"
say "Done! 37,840 training questions and 4,568 test questions are ready."
say "Let's peek at ONE training example: state + question + options -> one letter."
run "head -n 1 tev1/data/new-v1/records/dev.jsonl | python3 -m json.tool | head -40"
