"""
Configuration for different model backends (OpenAI, vLLM, HuggingFace)
"""
import os

# Model configurations
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4o": {
        "backend": "openai",
        "model_name": "gpt-4o-2024-05-13",
        "api_key_env": "OPENAI_API_KEY"
    },
    
    # vLLM-served models (OpenAI-compatible API)
    "qwen2.5-7b": {
        "backend": "vllm",
        "model_name": "/research/projects/mllab/public_llms/Qwen2.5-7B-Instruct",
        "base_url": "http://localhost:8000/v1",  # vLLM server URL (OpenAI client adds /chat/completions)
        "model_path": "/research/projects/mllab/public_llms/Qwen2.5-7B-Instruct"
    },
    "qwen3": {
        "backend": "vllm",
        "model_name": "/research/projects/mllab/public_llms/Qwen3",
        "base_url": "http://localhost:8000/v1",
        "model_path": "/research/projects/mllab/public_llms/Qwen3"
    },
    "deepseek-r1": {
        "backend": "vllm",
        "model_name": "/research/projects/mllab/public_llms/deepseek-r1",
        "base_url": "http://localhost:8000/v1",
        "model_path": "/research/projects/mllab/public_llms/deepseek-r1"
    },
    "qwen2.5-3b": {
        "backend": "vllm",
        "model_name": "/research/projects/mllab/public_llms/Qwen2.5-3B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "model_path": "/research/projects/mllab/public_llms/Qwen2.5-3B-Instruct"
    },
    "unsloth-llama-3.1-8b": {
        "backend": "vllm",
        "model_name": "/research/projects/mllab/public_llms/unsloth-Meta-Llama-3.1-8B",
        "base_url": "http://localhost:8002/v1",
        "model_path": "/research/projects/mllab/public_llms/unsloth-Meta-Llama-3.1-8B",
        "api_mode": "completions",
        "completion_prompt_format": "llama3_instruct",
        "completion_stop": ["<|eot_id|>"]
    },
    
    # Direct HuggingFace loading (slower, but no server needed)
    "qwen2.5-7b-direct": {
        "backend": "huggingface",
        "model_name": "Qwen2.5-7B-Instruct",
        "model_path": "/research/projects/mllab/public_llms/Qwen2.5-7B-Instruct"
    },
    "unsloth-llama-3.1-8b-direct": {
        "backend": "huggingface",
        "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "model_path": "/research/projects/mllab/public_llms/unsloth-Meta-Llama-3.1-8B"
    }
}

def get_model_config(model_key):
    """Get configuration for a specific model"""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_key]
