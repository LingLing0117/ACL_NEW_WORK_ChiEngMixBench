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
Script to build the Minimal Contrastive Pairs (MCP) dataset using LLM APIs.
This script reads raw text, identifies potential term-switching candidates,
and queries an LLM to generate the contrastive counterpart.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time
import argparse
import logging
from typing import List, Dict, Optional

# Third-party libraries (ensure these are in requirements.txt)
import google.generativeai as genai
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class MinimalPair(BaseModel):
    """Schema for structured LLM output."""
    valid: bool = Field(description="Whether the sentence contains a valid technical term code-mix.")
    target_term: str = Field(description="The specific English technical term found.")
    topic: str = Field(description="The technical sub-field (e.g., NLP, CV, RL).")
    prefix: str = Field(description="A short context prefix to set the scene.")
    sentence_A: str = Field(description="The original sentence with the English term.")
    sentence_B: str = Field(description="The contrastive sentence with the Chinese translation of the term.")
    is_hardcore: bool = Field(description="True if the term is highly specialized (Level-1).")


def is_contain_english(text: str) -> bool:
    """Checks if the text contains basic ASCII characters."""
    return any(char.isascii() and char.isalpha() for char in text)


def process_sentence_with_llm(client: genai.Client, model_name: str, text: str) -> Optional[Dict]:
    """
    Sends a text segment to the LLM to generate a Minimal Contrastive Pair.
    
    Args:
        client: The initialized GenAI client.
        model_name: The model identifier.
        text: The raw input text string.
        
    Returns:
        A dictionary containing the structured MCP data, or None if invalid.
    """
    prompt = f"""
    Task: Analyze the following text from a Chinese technical community.
    1. Identify if it contains a specific English technical term mixed with Chinese syntax.
    2. If yes, construct a Minimal Contrastive Pair (MCP):
       - Sentence A: Keep the English term.
       - Sentence B: Replace the English term with its standard Chinese academic translation.
       - Keep the rest of the sentence IDENTICAL.
    
    Input Text: "{text}"
    
    Output format must follow the JSON schema provided.
    """
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': MinimalPair,
            },
        )
        return json.loads(response.text)
    except Exception as e:
        logger.warning(f"API call failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="The input raw data file (e.g., .jsonl or .txt).")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The output file path for processed MCP pairs.")
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY", type=str,
                        help="The environment variable name for the API key.")
    parser.add_argument("--model_name", default="gemini-2.0-flash", type=str,
                        help="The model name to use for generation.")
    parser.add_argument("--max_samples", default=-1, type=int,
                        help="Limit the number of samples to process. -1 for all.")

    args = parser.parse_args()

    # 1. Setup API Client
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"API Key not found. Please set the environment variable '{args.api_key_env}'.")
    
    client = genai.Client(api_key=api_key)
    logger.info(f"Initialized GenAI client with model: {args.model_name}")

    # 2. Read Input Data
    processed_data = []
    processed_count = 0
    
    logger.info(f"Reading from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    logger.info(f"Total lines found: {total_lines}")

    # 3. Processing Loop
    for i, line in enumerate(lines):
        if args.max_samples != -1 and processed_count >= args.max_samples:
            break
            
        line = line.strip()
        if not line:
            continue

        # Robust reading logic for JSONL or plain text
        try:
            data_obj = json.loads(line)
            # Try to fetch content from various common keys
            text = data_obj.get('content') or data_obj.get('text') or list(data_obj.values())[0]
        except json.JSONDecodeError:
            text = line
            
        # Basic filtering
        if len(text) < 10 or not is_contain_english(text):
            continue

        logger.info(f"Processing line {i+1}/{total_lines}: {text[:30]}...")

        # LLM Processing
        result = process_sentence_with_llm(client, args.model_name, text)
        
        if result and result.get("valid") is True:
            # Add metadata
            result['source_id'] = i
            processed_data.append(result)
            processed_count += 1
            
            # Periodic saving
            if processed_count % 10 == 0:
                with open(args.output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(processed_data, out_f, ensure_ascii=False, indent=2)
                logger.info(f"Saved checkpoint: {processed_count} samples.")

    # Final Save
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        json.dump(processed_data, out_f, ensure_ascii=False, indent=2)
    
    logger.info(f"Job complete. Total valid MCP pairs saved: {len(processed_data)}")


if __name__ == "__main__":
    main()