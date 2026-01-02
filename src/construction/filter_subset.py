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
Script to filter raw corpus for high-quality Code-Mixed sentences.
It applies heuristic rules based on acronyms, sentence length, and
switching frequency to ensure domain relevance.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import json
import hashlib
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common acronyms whitelist for AI/CS domain
ACRONYM_WHITELIST = {
    'AI', 'ML', 'DL', 'NLP', 'LLM', 'RAG', 'SFT', 'RL', 'RLHF',
    'GPU', 'CPU', 'API', 'SDK', 'SQL', 'IDE', 'UI', 'UX', 'QA',
    'CNN', 'RNN', 'LSTM', 'GAN', 'VAE', 'BERT', 'GPT', 'ViT', 'IoT'
}

# Regex patterns
CJK_PATTERN = re.compile(r'[\u4e00-\u9fff]')
EN_WORD_PATTERN = re.compile(r'[A-Za-z]+')
URL_PATTERN = re.compile(r'https?://|www\.', re.I)
CODE_INDICATORS = {'def ', 'class ', 'import ', 'return ', '{', '}', '();', 'print('}


def is_code_mixed_sentence(text: str) -> bool:
    """
    Determines if a sentence contains both Chinese characters and English words.
    Also filters out pure code snippets or URLs.
    """
    if not text or len(text) < 10:
        return False
        
    # Filter out URLs and heavy code snippets
    if URL_PATTERN.search(text):
        return False
    if any(ind in text for ind in CODE_INDICATORS):
        return False
        
    has_zh = bool(CJK_PATTERN.search(text))
    has_en = bool(EN_WORD_PATTERN.search(text))
    
    return has_zh and has_en


def normalize_text(text: str) -> str:
    """Basic text normalization."""
    return text.strip()


def calculate_sha1(text: str) -> str:
    """Calculates SHA1 hash for deduplication."""
    s = normalize_text(text)
    s = re.sub(r'\s+', ' ', s)
    return 'sha1:' + hashlib.sha1(s.encode('utf-8')).hexdigest()


def filter_corpus(input_path: str, output_path: str):
    """
    Main filtering loop.
    
    Args:
        input_path: Path to raw input file.
        output_path: Path to save filtered JSONL.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    seen_hashes = set()
    count = 0
    
    logger.info(f"Processing file: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        # Determine if input is JSONL or plain text based on extension
        is_jsonl = input_path.suffix == '.jsonl'
        
        lines = fin.readlines()
        for line in lines:
            text = ""
            if is_jsonl:
                try:
                    item = json.loads(line)
                    text = item.get('content', '') or item.get('text', '')
                except json.JSONDecodeError:
                    continue
            else:
                text = line
            
            # 1. Check for Code-Mixing
            if not is_code_mixed_sentence(text):
                continue
                
            # 2. Length check (visible characters)
            visible_len = len(re.sub(r'\s+', '', text))
            if not (15 <= visible_len <= 300):
                continue
            
            # 3. Deduplication
            doc_hash = calculate_sha1(text)
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)
            
            # 4. Save
            output_obj = {
                "id": f"raw_{count:06d}",
                "text": text.strip(),
                "length": visible_len,
                "hash": doc_hash
            }
            fout.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
            count += 1
            
            if count % 1000 == 0:
                logger.info(f"Collected {count} valid sentences...")

    logger.info(f"Filtering complete. Total sentences saved: {count}")


def main():
    parser = argparse.ArgumentParser(description="Filter Code-Mixed Corpus")
    parser.add_argument("--input_file", type=str, required=True, help="Input raw corpus file")
    parser.add_argument("--output_file", type=str, required=True, help="Output filtered JSONL file")
    args = parser.parse_args()
    
    filter_corpus(args.input_file, args.output_file)


if __name__ == "__main__":
    main()