#!/usr/bin/env bash
# Reproduce every number in README.md and the report. Uses GPUs 0-3 (edit GPUS if needed).
# Approximate wall-clock on 4x H200: features 10 min, SDXL 15 min, methods 4 min, LoRA 12 min.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPUS=(0 1 2 3)
MODELS=(siglip2_so400m siglip2_g siglip_so400m clip_l14 dfn_h14 pe_g14)

# 0. One-shot draws (deterministic; the committed files also store each image's SHA-256)
$PY -m src.data

# 1. Frozen features: test views, shot views, prompts (one model per GPU, round robin)
for i in "${!MODELS[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  $PY extract_features.py --model "${MODELS[$i]}" --device cuda:$g > "logs/extract_${MODELS[$i]}.log" 2>&1 &
  if (( (i + 1) % ${#GPUS[@]} == 0 )); then wait; fi
done
wait

# 2. SDXL support sets (text-to-image + IP-Adapter variations of each draw's shots), sharded over GPUs
for s in "${!GPUS[@]}"; do
  $PY generate.py --shard "$s" --nshards ${#GPUS[@]} --device cuda:${GPUS[$s]} > "logs/generate_$s.log" 2>&1 &
done
wait
for i in "${!MODELS[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  $PY extract_synth.py --model "${MODELS[$i]}" --device cuda:$g > "logs/synth_${MODELS[$i]}.log" 2>&1 &
  if (( (i + 1) % ${#GPUS[@]} == 0 )); then wait; fi
done
wait

# 3. All classifiers on cached features, 10 draws -> results/acc_<model>.csv
$PY run_methods.py --device cuda:${GPUS[0]} > logs/run_methods.log 2>&1

# 4. CLIP-LoRA fine-tuning of the main vision encoder, one run per draw (about 4 min each on an H200)
for k in $(seq 0 9); do
  g=${GPUS[$((k % ${#GPUS[@]}))]}
  $PY finetune_lora.py --seed "$k" --device cuda:$g > "logs/lora_seed$k.log" 2>&1 &
  if (( (k + 1) % ${#GPUS[@]} == 0 )); then wait; fi
done
wait
$PY eval_lora.py
# prompt sets and CLAP anchor on the main backbone (also uses the LoRA features)
$PY run_ablations.py --device cuda:${GPUS[0]} > logs/run_ablations.log 2>&1
$PY scripts/summarize.py
$PY scripts/oracle_probe.py

# 5. Final model from the 16 raw images of draw 0, evaluated from the raw test images
$PY train.py --model siglip2_so400m --seed 0
$PY evaluate.py checkpoints/siglip2_so400m_seed0.pt --per-class
