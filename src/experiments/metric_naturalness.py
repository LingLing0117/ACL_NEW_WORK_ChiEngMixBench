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
Naturalness Metric Calculator (Expert Deviation Penalty).

This script implements the automated scoring system that penalizes generated
responses based on their deviation from expert norms in three dimensions:
1. Semantic Style (Embedding distance from expert centroid)
2. Structural Constraints (English ratio & Switching frequency)
3. Terminological Compactness (Acronym usage correctness)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re
import sys
import argparse
import logging
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Add parent directory to sys.path to allow importing from src.utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
try:
    from src.utils.term_rules import ACRONYM_RULES
except ImportError:
    ACRONYM_RULES = {}
    logging.warning("Could not import ACRONYM_RULES. Terminology penalty will be skipped.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NaturalnessScorer:
    def __init__(self, config_path, device="cpu"):
        self.device = device
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.std = self.config['standards']
        self.pen = self.config['penalties']
        
        # Load Semantic Model
        model_name = self.config['meta'].get('model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
        logger.info(f"Loading semantic model: {model_name}...")
        self.sim_model = SentenceTransformer(model_name, device=device)
        self.golden_vector = torch.tensor(self.std['golden_vector']).to(device)

    def _analyze_structure(self, text):
        en_tokens = re.findall(r'[a-zA-Z]+', text)
        zh_tokens = re.findall(r'[\u4e00-\u9fa5]', text)
        total = len(en_tokens) + len(zh_tokens)
        if total == 0: return 0.0, 0.0
        
        ratio = len(en_tokens) / total
        
        # Simplified switch frequency
        tokens = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fa5]', text)
        switches = 0
        if len(tokens) > 1:
            types = [1 if re.match(r'[a-zA-Z]+', t) else 0 for t in tokens]
            for i in range(1, len(types)):
                if types[i] != types[i-1]:
                    switches += 1
        freq = switches / total
        return ratio, freq

    def _check_terms(self, text):
        penalty_score = 0.0
        for acronym, variants in ACRONYM_RULES.items():
            # If standard acronym is present, no penalty
            if re.search(r'\b' + re.escape(acronym) + r'\b', text):
                continue
                
            # Check for non-standard variants (Full name or Chinese translation)
            for v in variants:
                if v in text:
                    penalty_score += self.pen['term_single']
                    break
        return min(penalty_score, self.pen['term_cap'])

    def score(self, text):
        deductions = {'ratio': 0.0, 'switch': 0.0, 'emd': 0.0, 'term': 0.0}
        
        # 1. Structural Penalty
        ratio, freq = self._analyze_structure(text)
        
        # Ratio Penalty (Range check)
        r_min, r_max = self.std['ratio_range']
        if ratio < r_min:
            deductions['ratio'] = (r_min - ratio) * self.pen['ratio_k']
        elif ratio > r_max:
            deductions['ratio'] = (ratio - r_max) * self.pen['ratio_k']
            
        # Switch Penalty (Upper bound check)
        s_limit = self.std['switch_limit']
        if freq > s_limit:
            deductions['switch'] = (freq - s_limit) * self.pen['switch_k']

        # 2. Semantic Penalty
        curr_emb = self.sim_model.encode(text, convert_to_tensor=True, device=self.device)
        dist = 1.0 - util.cos_sim(curr_emb, self.golden_vector).item()
        
        thresh = self.std['dist_threshold']
        if dist > thresh:
            deductions['emd'] = (dist - thresh) * self.pen['dist_k2']
        else:
            deductions['emd'] = dist * self.pen['dist_k1']

        # 3. Term Penalty
        deductions['term'] = self._check_terms(text)

        # Final Score
        total_penalty = sum(deductions.values())
        final_score = max(0.0, 5.0 - total_penalty)
        
        return final_score, deductions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Model output JSON file.")
    parser.add_argument("--config_file", type=str, default="config/scoring_config.json", help="Scoring configuration.")
    parser.add_argument("--output_file", type=str, required=True, help="Result JSON file.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = NaturalnessScorer(args.config_file, device)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    logger.info("Scoring responses...")
    
    for item in data:
        text = item.get('response', '') or item.get('content', '')
        if not text: continue
        
        score, penalties = scorer.score(text)
        
        item['naturalness_score'] = score
        item['penalties'] = penalties
        results.append(item)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    avg_score = np.mean([r['naturalness_score'] for r in results])
    logger.info(f"Average Naturalness Score: {avg_score:.2f}")
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()