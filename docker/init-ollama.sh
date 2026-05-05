#!/bin/sh
# Starts the Ollama server and pulls the target model on first run.
# Model is cached in the ollama_data volume so subsequent starts are instant.
set -e

MODEL="${OLLAMA_MODEL:-qwen3:8b}"

ollama serve &
SERVER_PID=$!

echo "[ollama] Waiting for server..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 2
done
echo "[ollama] Server ready."

if ollama list | grep -q "^${MODEL}"; then
  echo "[ollama] Model ${MODEL} already present."
else
  echo "[ollama] Pulling ${MODEL} — this may take several minutes on first run..."
  ollama pull "$MODEL"
  echo "[ollama] Model ready."
fi

wait "$SERVER_PID"
