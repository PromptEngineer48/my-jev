# Helpers sourced inside asciinema recordings: human-speed typing, real execution.
# Commands are REAL; only the typing animation is simulated.
cd /workspace/my-jev
export HF_HOME=/workspace/hf_cache TERM=xterm-256color PS1='\[\e[1;32m\]runpod\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]$ '
prompt() { printf '\e[1;32mrunpod\e[0m:\e[1;34m%s\e[0m$ ' "${PWD/#\/workspace\//}"; }
say() {  # beginner-friendly comment line
  printf '\e[1;33m# %s\e[0m\n' "$*"; sleep 1.2
}
typeit() {
  local s="$1" i
  for ((i = 0; i < ${#s}; i++)); do printf '%s' "${s:i:1}"; sleep 0.0$((RANDOM % 4 + 2)); done
  sleep 0.4; printf '\n'
}
run() { prompt; typeit "$1"; eval "$1"; sleep 1.5; }
