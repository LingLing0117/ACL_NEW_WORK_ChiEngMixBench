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
Spontaneity Metric Calculator (EPR & CSG).

This script calculates the probabilistic preference of LLMs for Code-Mixing
using Minimal Contrastive Pairs (MCP). It computes:
1. English Preference Rate (EPR): % of samples where English term has higher likelihood.
2. Contextual Spontaneity Gap (CSG): The log-likelihood difference (En - Zh).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_sentence_score(model, tokenizer, sentence, device):
    """
    Calculates the Length-Normalized Log-Likelihood of a sentence.
    
    Equation: S(t|C) = (1/L) * sum(log P(x_i | C, x_<i))
    Note: We use the negative CrossEntropyLoss provided by PyTorch, 
    which effectively gives the mean log-likelihood per token.
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # CrossEntropyLoss is -log(P), so we negate it to get log(P)
        # It is already averaged by sequence length (reduction='mean' default)
        score = -outputs.loss.item()
    
    return score


def main():
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model (e.g., 'Qwen/Qwen2.5-7B').")
    parser.add_argument("--data_file", default=None, type=str, required=True,
                        help="Path to the MCP dataset JSON file.")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="Path to save the evaluation results JSON.")
    
    ## Optional parameters
    parser.add_argument("--device", default="auto", type=str,
                        help="Device to use: 'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--quantization", action="store_true",
                        help="Whether to load model in 4-bit/8-bit quantization (requires bitsandbytes).")

    args = parser.parse_args()

    # 1. Setup Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # 2. Load Model & Tokenizer
    logger.info(f"Loading model from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        load_kwargs = {"trust_remote_code": True, "device_map": device}
        if args.quantization:
            load_kwargs["load_in_4bit"] = True
            
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 3. Load Data
    with open(args.data_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} MCP pairs.")

    # 4. Evaluation Loop
    results = []
    stats = {
        'hardcore': {'count': 0, 'win_en': 0, 'csg_list': []},
        'general': {'count': 0, 'win_en': 0, 'csg_list': []}
    }

    for item in tqdm(dataset, desc="Evaluating"):
        sent_en = item['sentence_A']  # Mixed (English term)
        sent_zh = item['sentence_B']  # Monolingual (Chinese term)
        is_hardcore = item.get('is_hardcore', False)
        
        # Calculate scores (Log-Likelihood)
        score_en = calculate_sentence_score(model, tokenizer, sent_en, device)
        score_zh = calculate_sentence_score(model, tokenizer, sent_zh, device)
        
        # Metrics
        csg = score_en - score_zh  # Contextual Spontaneity Gap
        win_en = 1 if score_en > score_zh else 0
        
        # Update Stats
        group_key = 'hardcore' if is_hardcore else 'general'
        stats[group_key]['count'] += 1
        stats[group_key]['win_en'] += win_en
        stats[group_key]['csg_list'].append(csg)
        
        results.append({
            "id": item.get('id'),
            "term": item.get('target_term'),
            "is_hardcore": is_hardcore,
            "score_a": score_en,
            "score_b": score_zh,
            "csg": csg
        })

    # 5. Save Results
    # Create directory if not exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(f"Raw results saved to {args.output_file}")

    # 6. Print Summary Report
    def get_metrics(s):
        count = s['count']
        if count == 0: return 0.0, 0.0
        epr = (s['win_en'] / count) * 100
        avg_csg = np.mean(s['csg_list'])
        return epr, avg_csg

    epr_hard, csg_hard = get_metrics(stats['hardcore'])
    epr_gen, csg_gen = get_metrics(stats['general'])

    print("\n" + "="*60)
    print(f"📊 Spontaneity Evaluation Report | Model: {os.path.basename(args.model_path)}")
    print("="*60)
    print(f"{'Group':<20} | {'Count':<8} | {'EPR (%)':<10} | {'Avg CSG':<10}")
    print("-" * 60)
    print(f"{'Specialized Terms':<20} | {stats['hardcore']['count']:<8} | {epr_hard:<10.2f} | {csg_hard:<10.4f}")
    print(f"{'General Terms':<20} | {stats['general']['count']:<8} | {epr_gen:<10.2f} | {csg_gen:<10.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()