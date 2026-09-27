source /workspace/my-jev/recording/demo.sh
say "Step 5: serve OUR model straight from Hugging Face with vLLM."
run "cat serve.sh"
say "Starting the server in the background. It downloads the model and loads it on the GPU."
run "nohup bash serve.sh > serve.log 2>&1 &"
say "Waiting until the server says it is healthy..."
run "until curl -s localhost:8000/health >/dev/null; do sleep 5; printf '.'; done; echo ' READY'"
run "tail -n 3 serve.log"
