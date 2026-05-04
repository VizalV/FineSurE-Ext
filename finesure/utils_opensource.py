"""
Extended utilities for FineSurE with support for open-source models
"""
import ast
import os
import re
import time
from typing import Optional, Dict, Any
from model_config import get_model_config, MODEL_CONFIGS

ERROR_TYPES = ['out-of-context error', 'entity error', 'predicate error', 'circumstantial error', 
               'grammatical error', 'coreference error', 'linking error', 'other error']


def _extract_expected_sentence_count(prompt: str) -> Optional[int]:
    patterns = [
        r"Summary with\s+(\d+)\s+sentences:\s*$",
        r"Summary\s*\((\d+)\s+sentences\):\s*$",
        r"Not-entailed summary sentences\s*\((\d+)\):\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1))
    return None


def _extract_expected_keyfact_count(prompt: str) -> Optional[int]:
    match = re.search(r"(\d+)\s+key\s+facts:\s*$", prompt, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        return int(match.group(1))
    return None


def _is_keyfact_prompt(prompt: str) -> bool:
    lower_prompt = prompt.lower()
    return ("key facts" in lower_prompt and "line number" in lower_prompt) or '"key fact"' in lower_prompt


def _is_entailment_prompt(prompt: str) -> bool:
    lower_prompt = prompt.lower()
    return ('"entailment"' in lower_prompt) or ("entailed|not_entailed" in lower_prompt)


def _is_error_subtype_prompt(prompt: str) -> bool:
    lower_prompt = prompt.lower()
    return ("error subtype classifier" in lower_prompt) or ("already known to be not entailed" in lower_prompt)


def _build_guided_factcheck_schema(expected_count: Optional[int]) -> Dict[str, Any]:
    categories = [
        "no error",
        "out-of-context error",
        "entity error",
        "predicate error",
        "circumstantial error",
        "grammatical error",
        "coreference error",
        "linking error",
        "other error",
    ]

    item_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "sentence": {"type": "string"},
            "reason": {"type": "string"},
            "category": {"type": "string", "enum": categories},
        },
        "required": ["sentence", "reason", "category"],
        "additionalProperties": False,
    }

    array_schema: Dict[str, Any] = {
        "type": "array",
        "items": item_schema,
    }

    if expected_count is not None:
        array_schema["minItems"] = expected_count
        array_schema["maxItems"] = expected_count

    return array_schema


def _build_guided_entailment_schema(expected_count: Optional[int]) -> Dict[str, Any]:
    item_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "sentence": {"type": "string"},
            "entailment": {"type": "string", "enum": ["entailed", "not_entailed"]},
            "reason": {"type": "string"},
        },
        "required": ["sentence", "entailment", "reason"],
        "additionalProperties": False,
    }

    array_schema: Dict[str, Any] = {
        "type": "array",
        "items": item_schema,
    }

    if expected_count is not None:
        array_schema["minItems"] = expected_count
        array_schema["maxItems"] = expected_count

    return array_schema


def _build_guided_subtype_schema(expected_count: Optional[int]) -> Dict[str, Any]:
    categories = [
        "out-of-context error",
        "entity error",
        "predicate error",
        "circumstantial error",
        "grammatical error",
        "coreference error",
        "linking error",
        "other error",
    ]

    item_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "sentence": {"type": "string"},
            "category": {"type": "string", "enum": categories},
            "reason": {"type": "string"},
        },
        "required": ["sentence", "category", "reason"],
        "additionalProperties": False,
    }

    array_schema: Dict[str, Any] = {
        "type": "array",
        "items": item_schema,
    }

    if expected_count is not None:
        array_schema["minItems"] = expected_count
        array_schema["maxItems"] = expected_count

    return array_schema


def _build_guided_keyfact_schema(expected_count: Optional[int]) -> Dict[str, Any]:
    item_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "key fact": {"type": "string"},
            "response": {"type": "string", "enum": ["Yes", "No"]},
            "line number": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["key fact", "response", "line number"],
        "additionalProperties": False,
    }

    array_schema: Dict[str, Any] = {
        "type": "array",
        "items": item_schema,
    }

    if expected_count is not None:
        array_schema["minItems"] = expected_count
        array_schema["maxItems"] = expected_count

    return array_schema


def _build_guided_schema_for_prompt(prompt: str) -> Dict[str, Any]:
    if _is_keyfact_prompt(prompt):
        return _build_guided_keyfact_schema(_extract_expected_keyfact_count(prompt))
    if _is_entailment_prompt(prompt):
        return _build_guided_entailment_schema(_extract_expected_sentence_count(prompt))
    if _is_error_subtype_prompt(prompt):
        return _build_guided_subtype_schema(_extract_expected_sentence_count(prompt))
    return _build_guided_factcheck_schema(_extract_expected_sentence_count(prompt))


def _build_llama_strict_keyfact_prompt(prompt: str) -> str:
    expected_count = _extract_expected_keyfact_count(prompt)
    count_rule = ""
    if expected_count is not None:
        count_rule = f"The list must contain exactly {expected_count} objects.\n"

    strict_block = (
        "\n\nMANDATORY OUTPUT FORMAT:\n"
        "Return ONLY one JSON array. Do not output any prose, explanation, or markdown.\n"
        + count_rule
        + "Each array item must be an object with exactly these keys: \"key fact\", \"response\", \"line number\".\n"
        + "\"response\" must be exactly \"Yes\" or \"No\".\n"
        + "\"line number\" must be a JSON array of integers (empty array allowed).\n"
        + "If you fail to follow this format, the output will be discarded.\n"
    )
    return prompt + strict_block


def _build_llama_strict_entailment_prompt(prompt: str) -> str:
    expected_count = _extract_expected_sentence_count(prompt)
    count_rule = ""
    if expected_count is not None:
        count_rule = f"The list must contain exactly {expected_count} objects.\n"

    strict_block = (
        "\n\nMANDATORY OUTPUT FORMAT:\n"
        "Return ONLY one JSON array. Do not output any prose, explanation, or markdown.\n"
        + count_rule
        + "Each array item must be an object with exactly these keys: \"sentence\", \"entailment\", \"reason\".\n"
        + "\"entailment\" must be exactly \"entailed\" or \"not_entailed\".\n"
        + "Keep reason brief (max 20 words).\n"
        + "If you fail to follow this format, the output will be discarded.\n"
    )
    return prompt + strict_block


def _build_llama_strict_subtype_prompt(prompt: str) -> str:
    expected_count = _extract_expected_sentence_count(prompt)
    count_rule = ""
    if expected_count is not None:
        count_rule = f"The list must contain exactly {expected_count} objects.\n"

    strict_block = (
        "\n\nMANDATORY OUTPUT FORMAT:\n"
        "Return ONLY one JSON array. Do not output any prose, explanation, or markdown.\n"
        + count_rule
        + "Each array item must be an object with exactly these keys: \"sentence\", \"category\", \"reason\".\n"
        + "\"category\" must be exactly one of: \"out-of-context error\", \"entity error\", \"predicate error\", \"circumstantial error\", \"grammatical error\", \"coreference error\", \"linking error\", \"other error\".\n"
        + "Keep reason brief (max 20 words).\n"
        + "If you fail to follow this format, the output will be discarded.\n"
    )
    return prompt + strict_block


def _build_llama_strict_prompt(prompt: str) -> str:
    expected_count = _extract_expected_sentence_count(prompt)
    count_rule = ""
    if expected_count is not None:
        count_rule = f"The list must contain exactly {expected_count} objects.\n"

    strict_block = (
        "\n\nMANDATORY OUTPUT FORMAT:\n"
        "Return ONLY one JSON array. Do not output any prose, explanation, or markdown.\n"
        + count_rule
        + "Each array item must be an object with exactly these keys: \"sentence\", \"reason\", \"category\".\n"
        + "\"category\" must be exactly one of: \"no error\", \"out-of-context error\", \"entity error\", \"predicate error\", \"circumstantial error\", \"grammatical error\", \"coreference error\", \"linking error\", \"other error\".\n"
        + "If you fail to follow this format, the output will be discarded.\n"
    )
    return prompt + strict_block


def _build_completion_prompt(prompt: str, config: Dict) -> str:
    """Build completion-style prompt for models that need explicit instruct formatting."""
    prompt_format = config.get("completion_prompt_format", "plain")
    if prompt_format == "llama3_instruct":
        if _is_keyfact_prompt(prompt):
            strict_prompt = _build_llama_strict_keyfact_prompt(prompt)
        elif _is_entailment_prompt(prompt):
            strict_prompt = _build_llama_strict_entailment_prompt(prompt)
        elif _is_error_subtype_prompt(prompt):
            strict_prompt = _build_llama_strict_subtype_prompt(prompt)
        else:
            strict_prompt = _build_llama_strict_prompt(prompt)
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + strict_prompt
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return prompt


def get_response_opensource(prompt: str, model_key: str = "qwen2.5-7b", temperature: float = 0.0, 
                           max_tokens: int = 2048, **kwargs) -> str:
    """
    Unified function to get response from any model backend
    
    Args:
        prompt: Input prompt
        model_key: Key from MODEL_CONFIGS (e.g., "qwen2.5-7b", "gpt-4o")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional arguments for specific backends
        
    Returns:
        text_response: The output from the model
    """
    config = get_model_config(model_key)
    backend = config["backend"]
    
    if backend == "openai":
        return _get_response_openai(prompt, config, temperature, max_tokens)
    elif backend == "vllm":
        return _get_response_vllm(prompt, config, temperature, max_tokens)
    elif backend == "huggingface":
        return _get_response_huggingface(prompt, config, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _get_response_openai(prompt: str, config: Dict, temperature: float, max_tokens: int) -> str:
    """Get response from OpenAI API"""
    import openai
    
    api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
    if not api_key:
        raise ValueError(f"API key not found in environment: {config.get('api_key_env')}")
    
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def _get_response_vllm(prompt: str, config: Dict, temperature: float, max_tokens: int) -> str:
    """
    Get response from vLLM server (OpenAI-compatible API)
    
    Requirements:
        - vLLM server must be running
        - Start with: python -m vllm.entrypoints.openai.api_server --model <model_path> --port 8000
    """
    import openai
    
    # Use OpenAI client but point to vLLM server
    client = openai.OpenAI(
        base_url=config["base_url"],
        api_key="EMPTY"  # vLLM doesn't require real API key
    )
    
    api_mode = config.get("api_mode", "chat_completions")

    if api_mode == "chat_completions":
        response = client.chat.completions.create(
            model=config["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    if api_mode == "completions":
        completion_prompt = _build_completion_prompt(prompt, config)
        request_args = {
            "model": config["model_name"],
            "prompt": completion_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "echo": False,
        }
        stop_tokens = config.get("completion_stop")
        if stop_tokens is not None:
            request_args["stop"] = stop_tokens

        extra_body: Dict[str, Any] = {}

        min_tokens = config.get("min_tokens")
        if min_tokens is not None:
            extra_body["min_tokens"] = int(min_tokens)

        if config.get("guided_json", False):
            extra_body["guided_json"] = _build_guided_schema_for_prompt(prompt)

        if extra_body:
            request_args["extra_body"] = extra_body

        response = client.completions.create(**request_args)
        return response.choices[0].text

    raise ValueError(f"Unsupported vLLM api_mode: {api_mode}")


def _get_response_huggingface(prompt: str, config: Dict, temperature: float, max_tokens: int) -> str:
    """
    Get response by directly loading HuggingFace model
    
    Note: This is slower than vLLM but doesn't require a server
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    debug_enabled = os.getenv("FINESURE_HF_DEBUG", "1") == "1"

    def _dbg(msg: str) -> None:
        if debug_enabled:
            print(f"[HF-DEBUG] {msg}", flush=True)

    # Cache model in global scope to avoid reloading
    if not hasattr(_get_response_huggingface, 'model_cache'):
        _get_response_huggingface.model_cache = {}
    
    model_path = config["model_path"]
    
    # Load model if not cached
    if model_path not in _get_response_huggingface.model_cache:
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _dbg(f"Tokenizer loaded. pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

        strict_gpu_only = os.getenv("FINESURE_STRICT_GPU_ONLY", "1") == "1"
        if strict_gpu_only and not torch.cuda.is_available():
            raise RuntimeError(
                "FINESURE_STRICT_GPU_ONLY=1 but CUDA is unavailable. "
                "Disable strict mode or run on a GPU node."
            )

        preferred_dtype = torch.bfloat16
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            preferred_dtype = torch.float16

        if strict_gpu_only:
            # Force all weights onto GPU 0. If the model does not fit, fail fast.
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=preferred_dtype,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=preferred_dtype,
                device_map="auto",
                trust_remote_code=True,
            )

        if strict_gpu_only:
            hf_device_map = getattr(model, "hf_device_map", None)
            if isinstance(hf_device_map, dict):
                bad_placements = {
                    name: dev
                    for name, dev in hf_device_map.items()
                    if not (
                        (isinstance(dev, int) and dev >= 0)
                        or (isinstance(dev, str) and dev.startswith("cuda"))
                    )
                }
                if bad_placements:
                    raise RuntimeError(
                        "Strict GPU mode violation: some model weights were not on CUDA devices: "
                        f"{bad_placements}"
                    )
            else:
                first_param_device = next(model.parameters()).device
                if first_param_device.type != "cuda":
                    raise RuntimeError(
                        "Strict GPU mode violation: model parameters are not on CUDA. "
                        f"First parameter device: {first_param_device}"
                    )

        _get_response_huggingface.model_cache[model_path] = (model, tokenizer)
        print(f"Model loaded successfully!")
        _dbg(f"Model device map ready. model.device={getattr(model, 'device', 'unknown')}")
        _dbg(f"hf_device_map={getattr(model, 'hf_device_map', None)}")
    else:
        model, tokenizer = _get_response_huggingface.model_cache[model_path]
        _dbg("Using cached HF model/tokenizer")
    
    # Format as chat (no fallback by design so tokenizer/chat-template errors are surfaced).
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    _dbg(f"Prompt chars={len(prompt)}; rendered chat template chars={len(text)}")
    
    # Generate
    t0 = time.perf_counter()
    model_inputs = tokenizer([text], return_tensors="pt")
    _dbg(f"Tokenization done in {time.perf_counter() - t0:.2f}s; input tokens={model_inputs.input_ids.shape[-1]}")

    t1 = time.perf_counter()
    model_inputs = model_inputs.to(model.device)
    _dbg(f"Moved inputs to device in {time.perf_counter() - t1:.2f}s")
    _dbg(f"Input tensor device={model_inputs.input_ids.device}, model.device={model.device}")

    _dbg(
        "Starting generation with "
        f"max_new_tokens={max_tokens}, temperature={temperature}, do_sample={temperature > 0}"
    )
    t2 = time.perf_counter()
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _dbg(f"Generation finished in {time.perf_counter() - t2:.2f}s")
    
    # Decode only the generated part
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    gen_tokens = generated_ids[0].shape[-1] if generated_ids else 0
    _dbg(f"Generated token count={gen_tokens}")
    
    t3 = time.perf_counter()
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    _dbg(f"Decode finished in {time.perf_counter() - t3:.2f}s; response chars={len(response)}")
    return response


# Import all other functions from original utils.py
def get_fact_checking_prompt(input, sentences):
    """Define the input prompt for fact checking"""
    num_sentences = str(len(sentences))
    sentences = '\n'.join(sentences)

    prompt = \
"""
You will receive a transcript followed by a corresponding summary. Your task is to assess the factuality of each summary sentence across nine categories:
* no error: the statement aligns explicitly with the content of the transcript and is factually consistent with it.
* out-of-context error: the statement contains information not present in the transcript.
* entity error: the primary arguments (or their attributes) of the predicate are wrong.
* predicate error: the predicate in the summary statement is inconsistent with the transcript.
* circumstantial error: the additional information (like location or time) specifying the circumstance around a predicate is wrong.
* grammatical error: the grammar of the sentence is so wrong that it becomes meaningless.
* coreference error: a pronoun or reference with wrong or non-existing antecedent.
* linking error: error in how multiple statements are linked together in the discourse (for example temporal ordering or causal link).
* other error: the statement contains any factuality error which is not defined here.

Instruction:
First, compare each summary sentence with the transcript.
Second, provide a single sentence explaining which factuality error the sentence has.
Third, answer the classified error category for each sentence in the summary.

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "sentence", "reason", and "category":
[{"sentence": "first sentence", "reason": "your reason", "category": "no error"}, {"sentence": "second sentence", "reason": "your reason", "category": "out-of-context error"}, {"sentence": "third sentence", "reason": "your reason", "category": "entity error"},]

Transcript:
%s

Summary with %s sentences:
%s
""" % (input, num_sentences, sentences)

    return prompt


def parsing_llm_fact_checking_output(output):
    """Parse the output from LLMs based on heuristic rules"""
    try:
        output = output.replace('```json', '').replace('```', '')
        start_idx = output.find('[')

        if start_idx != -1:
            end_idx = output.rfind(']')
            output = output[start_idx:end_idx+1]
            output = output.replace('\n','')
            output = ast.literal_eval(output)

            pred_labels, pred_types = [], []
            for out in output:
                category = out["category"]
                category = category.replace('\n', '').replace('[', '').replace(']', '')
                if category.lower() == "no error":
                    pred_labels.append(0)
                else:
                    pred_labels.append(1)
                pred_types.append(category)
            return pred_labels, pred_types
        
        else:
            start_idx = output.find('{')
            end_idx = output.rfind('}')
            output = output[start_idx:end_idx+1]
            output = output.replace('\n','')
            output = ast.literal_eval(output)

            pred_labels, pred_types = [], []
            category = output["category"]
            category = category.replace('\n', '').replace('[', '').replace(']', '')
            if category.lower() == "no error":
                pred_labels.append(0)
            else:
                pred_labels.append(1)
            pred_types.append(category)
            return pred_labels, pred_types
        
    except Exception as e:
        try:
            if "category" not in output.lower():
                return [], []
            subseqs = output.split("category")

            def error_detection(subseq):
                detected = False
                for error_type in ERROR_TYPES:
                    if error_type in subseq:
                        detected = True
                        detected_type = error_type
                if detected:
                    return 1, error_type
                else:
                    return 0, "no error"
                
            pred_labels, pred_types = [], []
            for subseq in subseqs:
                error_label, error_type = error_detection(subseq)
                pred_labels.append(error_label)
                pred_types.append(error_type)
        
            return pred_labels, pred_types
        
        except Exception as e:
            print('parsing error:', e)
            return [], []


def get_keyfact_alighment_prompt(keyfacts, sentences):
    """Define the input prompt for keyfact alignment"""
    summary = ['[' + str(line_num + 1) + '] ' + sentence for line_num, sentence in enumerate(sentences)]
    summary = '\n'.join(summary)
    num_key_facts = str(len(keyfacts))
    key_facts = '\n'.join(keyfacts)
    
    prompt = \
'''
You will receive a summary and a set of key facts for the same transcript. Your task is to assess if each key fact is inferred from the summary.

Instruction:
First, compare each key fact with the summary.
Second, check if the key fact is inferred from the summary and then response "Yes" or "No" for each key fact. If "Yes", specify the line number(s) of the summary sentence(s) relevant to each key fact. 

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "key fact", "response", and "line number":
[{"key fact": "first key fact", "response": "Yes", "line number": [1]}, {"key fact": "second key fact", "response": "No", "line number": []}, {"key fact": "third key fact", "response": "Yes", "line number": [1, 2, 3]}]

Summary:
%s

%s key facts:
%s
''' % (summary, num_key_facts, key_facts)

    return prompt


def parsing_llm_keyfact_alighment_output(output):
    """Parse the output from LLMs for keyfact alignment"""
    try:
        output = output.replace('```', '')
        start_idx = output.find('[')
        output = output[start_idx:]
        output = ast.literal_eval(output)

        matched_lines = set()
        pred_labels = []

        for out in output:
            category = out["response"]

            if category.lower() == "yes":
                pred_labels.append(1)
            else:
                pred_labels.append(0)
            
            if 'line number' in out:
                line_nums = out["line number"]

                for line_num in line_nums:
                    if type(line_num) is str:
                        line_num = line_num.replace('[', '').replace(']', '')
                    matched_lines.add(int(line_num))
        
        return pred_labels, list(matched_lines)
    
    except Exception as e:
        print(e)
        return [], []


def compute_faithfulness_percentage_score(pred_faithfulness_labels):
    """Compute faithfulness score"""
    faithfulness = 1.0 - sum(pred_faithfulness_labels) / len(pred_faithfulness_labels)  
    return faithfulness


def compute_completeness_percentage_score(pred_alignment_labels):
    """Compute completeness score"""
    completeness = sum(pred_alignment_labels) / len(pred_alignment_labels)  
    return completeness


def compute_conciseness_percentage_score(pred_sentence_line_numbers, num_sentences):
    """Compute conciseness score"""
    conciseness = len(pred_sentence_line_numbers) / num_sentences
    return conciseness
