source /workspace/my-jev/recording/demo.sh
say "Step 4: the real training run. All 37,840 examples, one epoch, then upload to Hugging Face."
run "bash train.sh"
say "Training finished and the model is on Hugging Face!"
run "cat outputs/my-jev-4b/train_report.json | python3 -m json.tool | head -50"
