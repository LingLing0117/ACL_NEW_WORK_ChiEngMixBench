ChiEngMixBench: Evaluating Spontaneous & Natural Code-Mixing 🇨🇳🇺🇸
<p align="center"> <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img src="https://www.google.com/search?q=https://img.shields.io/badge/Data%2520License-CC%2520BY--NC%25204.0-lightgrey"></a> <a href="https://www.google.com/search?q=LICENSE"><img src="https://www.google.com/search?q=https://img.shields.io/badge/Code%2520License-MIT-blue"></a> <a href="#"><img src="https://www.google.com/search?q=https://img.shields.io/badge/Python-3.8%252B-green"></a> </p>

This repository contains the dataset, source code, and evaluation logs for the paper "ChiEngMixBench: Evaluating Large Language Models on Spontaneous and Natural Chinese-English Code-Mixed Generation".

ChiEngMixBench is the first benchmark derived from authentic academic corpora to evaluate LLMs' code-mixing capabilities through two dimensions: Spontaneity (Cognitive Preference) and Naturalness (Pragmatic Compliance).

📂 Project Structure
The repository is organized as follows:

Plaintext

ChiEngMixBench/
│
├── data/
│   ├── raw/                 # Raw corpus samples (anonymized)
│   ├── processed/           # The core MCP benchmark dataset (JSON)
│   └── resources/           # Expert term banks and mapping rules
│
├── src/
│   ├── construction/        # Pipeline for dataset construction (Data mining & filtering)
│   ├── experiments/         # Implementation of evaluation metrics (EPR, CSG, Naturalness)
│   └── utils/               # Utility functions and static rule definitions
│
├── results/
│   ├── spontaneity/         # Raw logits probing logs (RQ1)
│   ├── naturalness/         # Naturalness scoring logs (RQ2)
│   └── human_eval/          # Anonymized human evaluation records
│
└── requirements.txt         # Python dependencies
🛠️ Installation & Setup
Environment: This project requires Python 3.8 or higher.

1. Install Dependencies:

Bash

pip install -r requirements.txt
Key libraries: torch, transformers, sentence-transformers, numpy, scikit-learn.

2. (Optional) API Key Configuration:

Note: You do NOT need an API key to reproduce the evaluation results on the provided dataset. An API key (e.g., Google Gemini / OpenAI) is only required if you intend to run the src/construction/ pipeline to build a new dataset from scratch.

Bash

export GEMINI_API_KEY="your_api_key_here"
🚀 Reproduction Guide
We provide one-click scripts to reproduce the main experimental results reported in the paper.

1. Reproducing Spontaneity Metrics (RQ1)
To calculate the English Preference Rate (EPR) and Contextual Spontaneity Gap (CSG):

Bash

# Run evaluation on the provided MCP dataset
python src/experiments/metric_spontaneity.py \
    --model_path "Qwen/Qwen2.5-7B-Instruct" \
    --data_file "data/processed/MCPdataset_cleaned_v1.json" \
    --output_file "results/reproduced_spontaneity.json"
Output will display the average CSG for Specialized vs. General terms.

2. Reproducing Naturalness Scores (RQ2)
To run the Expert Deviation Penalty (EDP) system on model outputs:

Bash

# Score the model responses
python src/experiments/metric_naturalness.py \
    --input_file "results/naturalness/Qwen2.5-7B-Instruct.json" \
    --config_file "src/experiments/config/scoring_config.json" \
    --output_file "results/reproduced_naturalness_scores.json"
Output will confirm the Naturalness Score (0-5 scale) and detailed penalty breakdown.

📊 Data Description
Core Benchmark (data/processed/MCPdataset_cleaned_v1.json)
The Minimal Contrastive Pairs (MCP) dataset follows this JSON schema:

JSON

{
  "id": 1024,
  "topic": "Deep Learning",
  "is_hardcore": true,
  "target_term": "Transformer",
  "sentence_A": "Transformer 的并行计算能力使其在 NLP 任务中表现优异。",
  "sentence_B": "变换器的并行计算能力使其在自然语言处理任务中表现优异。",
  "context_prefix": "在讨论模型架构时..."
}
sentence_A: The code-mixed sentence (Mixed).

sentence_B: The monolingual Chinese sentence (Mono).

is_hardcore: true if the term is a specialized Anchor Term (Specialized); false if it is a General Term.

Human Evaluation (results/human_eval/)
Contains the scoring records from our "3+1" expert team.

Score Range: 1.0 (Worst) to 5.0 (Best/Native).

Annotators: Names have been replaced with aliases (Evaluator_A, Evaluator_B, etc.) to protect privacy.

⚖️ License & Privacy
License
Code: Released under the MIT License.

Data: Released under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International).

Privacy & Compliance
This dataset is derived from publicly accessible technical communities. To protect user privacy and adhere to ethical research standards:

PII Scrubbing: All Personally Identifiable Information (PII), including user IDs, avatar URLs, and specific timestamps, has been rigorously removed.

Usage: This release is intended solely for academic research purposes. Commercial use of the scraped community data is strictly prohibited.

📜 Citation
If you find this work useful, please cite:

代码段

@article{chiengmixbench2026,
  title={ChiEngMixBench: Evaluating Large Language Models on Spontaneous and Natural Chinese-English Code-Mixed Generation},
  author={Yang, Qingyan and Wang, Tongxi and Luo, YunSheng},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
