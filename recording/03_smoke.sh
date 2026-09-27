source /workspace/my-jev/recording/demo.sh
say "Step 3: a quick smoke test. Train on just 200 examples to check everything works."
run "bash train.sh --max_train 200 --eval_n 90 --hf_repo ''"
