# grpo-alignment-sandbox

A lightweight reinforcement learning sandbox implementing Group Relative Policy Optimization (GRPO) to align open-source models (Qwen2.5-0.5B-Instruct) for structured mathematical reasoning.

- Base Architecture: Qwen-2.5-0.5B
- Hardware Target: Single NVIDIA T4 GPU (Google Colab Free Tier)
- Memory Optimization Stack: 8-bit AdamW, Gradient Checkpointing, LoRA ($r=8, \alpha=16$)
- Group Size ($G$): 2 generations per prompt

Through the 150-step sprint, the model rapidly transitioned its output probability distribution to accommodate custom programmatic regex rewards.

Sample:

<think>She earned $60</think> because 5 x $12 = $60.

The showed 'reward hacking', where the model adopted the target formatting constraints, all while the policy optimized for token-efficiency by consolidating the answer inside the initial reasoning boundary to minimize sequence length overhead.
