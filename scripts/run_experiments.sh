#!/bin/bash

# Configuration (based on recent logs, adjust as needed)
MODEL="qwen2.5-7b"
AGENTS=3
ROUNDS=3
DATA="gsm8k"
SIZE=50
AGENT_IDX=0 # First agent (0-indexed)

# Temperatures to traverse
TEMPS=(0.1 0.5 1.0 1.5 2.0 2.5)

# Experiment 1: Modify 1st Layer (Round 0 - Initial Opinions)
# echo "Starting Experiment 1: Modifying Round 0"
# for t in "${TEMPS[@]}"; do
#     echo "Running Round 0, Agent $AGENT_IDX, Temp $t"
#     python src/main.py --model $MODEL --num_agents $AGENTS --data $DATA --data_size $SIZE --debate_rounds $ROUNDS \
#         --target_round 0 --target_agent_idx $AGENT_IDX --target_temp $t
# done

# Experiment 2: Modify 2nd Layer (Round 1 - First Debate Round)
echo "Starting Experiment 2: Modifying Round 1"
for t in "${TEMPS[@]}"; do
    echo "Running Round 1, Agent $AGENT_IDX, Temp $t"
    python src/main.py --model $MODEL --num_agents $AGENTS --data $DATA --data_size $SIZE --debate_rounds $ROUNDS \
        --target_round 1 --target_agent_idx $AGENT_IDX --target_temp $t
done

# Experiment 3: Modify 3rd Layer (Round 2 - Second Debate Round)
echo "Starting Experiment 3: Modifying Round 2"
for t in "${TEMPS[@]}"; do
    echo "Running Round 2, Agent $AGENT_IDX, Temp $t"
    python src/main.py --model $MODEL --num_agents $AGENTS --data $DATA --data_size $SIZE --debate_rounds $ROUNDS \
        --target_round 2 --target_agent_idx $AGENT_IDX --target_temp $t
done

echo "All experiments completed."
