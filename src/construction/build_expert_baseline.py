# coding=utf-8
# Copyright 2025 The ChiEngMixBench Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Script to build the Expert Baseline Statistics.
It calculates the distribution of English token ratios, switching frequencies,
and semantic centroids from the high-quality expert corpus.
Output is a configuration JSON used by the Naturalness Scorer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re
import argparse
import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_structure_stats(text: str):
    """
    Analyzes structural features: English Token Ratio and Switching Frequency.
    """
    en_tokens = re.findall(r'[a-zA-Z]+', text)
    zh_tokens = re.findall(r'[\u4e00-\u9fa5]', text)
    
    num_en = len(en_tokens)
    num_zh = len(zh_tokens)
    total = num_en + num_zh
    
    if total == 0:
        return 0.0, 0.0
    
    # English Ratio
    ratio = num_en / total
    
    # Switching Frequency (simplified approximation)
    # Counts transitions between En and Zh blocks
    tokens_mixed = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fa5]', text)
    switches = 0
    if len(tokens_mixed) > 1:
        # 0 for Zh, 1 for En
        types = [1 if re.match(r'[a-zA-Z]+', t) else 0 for t in tokens_mixed]
        for i in range(1, len(types)):
            if types[i] != types[i-1]:
                switches += 1
                
    # Normalize switches by length (switches per token)
    switch_freq = switches / total
    
    return ratio, switch_freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, 
                        help="Path to the filtered expert corpus (JSONL).")
    parser.add_argument("--output_config", type=str, default="scoring_config.json",
                        help="Path to save the generated scoring configuration.")
    parser.add_argument("--model_name", type=str, default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="Sentence Transformer model for semantic centroid.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load Model
    logger.info(f"Loading SentenceTransformer: {args.model_name}...")
    sim_model = SentenceTransformer(args.model_name, device=device)

    # Containers
    ratios = []
    switch_freqs = []
    embeddings = []

    # 1. Iterate over data
    logger.info("Processing data file...")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    total_samples = 0
    for line in lines:
        try:
            item = json.loads(line)
            text = item.get('content') or item.get('text')
            if not text: 
                continue
                
            # Structure Stats
            r, s = calculate_structure_stats(text)
            ratios.append(r)
            switch_freqs.append(s)
            
            # Semantic Stats (Batch encoding is better, but doing line-by-line for simplicity here)
            # In production, use model.encode(batch)
            embeddings.append(sim_model.encode(text))
            
            total_samples += 1
            if total_samples % 100 == 0:
                logger.info(f"Processed {total_samples} samples...")
                
        except json.JSONDecodeError:
            continue
            
    # 2. Calculate Statistics
    logger.info("Calculating distribution statistics...")
    
    # Structural Thresholds
    mu_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    ratio_min = max(0.01, mu_ratio - 1.5 * std_ratio)
    ratio_max = min(0.95, mu_ratio + 2.0 * std_ratio)
    
    mu_switch = np.mean(switch_freqs)
    std_switch = np.std(switch_freqs)
    switch_limit = float(mu_switch + 1.2 * std_switch)
    
    # Semantic Centroid (Golden Vector)
    embeddings_np = np.array(embeddings)
    golden_vector = np.mean(embeddings_np, axis=0)
    
    # 3. Save Configuration
    config = {
        "meta": {
            "source_samples": total_samples,
            "model_name": args.model_name
        },
        "standards": {
            "ratio_range": [float(f"{ratio_min:.3f}"), float(f"{ratio_max:.3f}")],
            "switch_limit": float(f"{switch_limit:.3f}"),
            "golden_vector": golden_vector.tolist()
        },
        "penalties": {
            "term_single": 0.5,
            "term_cap": 2.0,
            "ratio_k": 2.5,
            "switch_k": 1.5,
            "dist_threshold": 0.65, 
            "dist_k1": 1.0,
            "dist_k2": 3.0
        }
    }
    
    with open(args.output_config, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        
    logger.info(f"Configuration saved to {args.output_config}")


if __name__ == "__main__":
    main()