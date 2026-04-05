"""
Presets for baseline and enhanced factuality categorization experiments.
"""

PRESETS = {
    "baseline": {
        "reasoning_mode": "single_stage",
        "prompt_style": "baseline",
        "n_samples": 1,
        "temperature": 0.0,
    },
    "enhanced": {
        "reasoning_mode": "two_stage",
        "prompt_style": "strict",
        "n_samples": 1,
        "temperature": 0.0,
    },
    "enhanced_sc": {
        "reasoning_mode": "two_stage",
        "prompt_style": "strict",
        "n_samples": 3,
        "temperature": 0.3,
    },
}


def get_preset(name):
    if name not in PRESETS:
        raise ValueError("Unknown preset: {}. Available: {}".format(name, list(PRESETS.keys())))
    return PRESETS[name].copy()
