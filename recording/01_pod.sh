source /workspace/my-jev/recording/demo.sh
say "Step 1: we are inside a RunPod GPU machine. Let's see which GPU we rented."
run "nvidia-smi --query-gpu=name,memory.total --format=csv"
say "An NVIDIA A40 with 48 GB of memory, about 49 cents an hour."
say "These are the files in our project folder."
run "ls"
say "Settings live in a .env file. Here is the template (never share your real keys!)"
run "cat .env.example"
