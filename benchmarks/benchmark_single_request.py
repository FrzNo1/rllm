"""
Benchmark the re-computation latency and memory of a single requests.
"""
import gc
import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams, LLMEngine, TokensPrompt
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.api_server import engine
from vllm.inputs import PromptType
from vllm.utils import FlexibleArgumentParser

def main():
	engine_args = EngineArgs(
		model="facebook/opt-125m",
	)
	my_engine = LLMEngine.from_engine_args(engine_args)

	sampling_params = SamplingParams(temperature=0)
	prompt_token_ids = TokensPrompt(prompt_token_ids=list(range(256)))

	my_engine.add_request(
		request_id=str(0),
		prompt=prompt_token_ids,
		params=sampling_params,
	)
	print(f"DEBUG >> Add request")

	outputs = engine.step()
	print(f"DEBUG >> finished")

if __name__ == '__main__':
	main()