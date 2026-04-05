# FineSurE: Fine-grained Summarization Evaluation using LLMs (ACL'24-main, Long Paper)

Here is our paper on arXiv: [[link](https://arxiv.org/abs/2407.00908)]

The structure of the projects:
- dataset: FRANK and REALSumm (in JSON format) are located in this folder
- reproduce: the code to reproduce the results of FineSurE in Table 1 and Table 2 
- finesure: the code to run our FineSurE method to evaluate the summary generated from language models

## Highlight

**FineSurE** is a *multi-dimensional*, *fine-grained* automated evaluation framework for text summarization. It covers there distinctive evaluation dimensions, namely faithfulness, completeness, and conciseness. These dimensions are crucial to assess the capability of modern language models in summarization, as they are susceptible to incorrect statement, information omission, and verbosity.

FineSurE framework breaks down a complicate evaluation process into ```two simple human-like evaluation tasks``` using LLMs. 

- **Fact Checking:** This is a task of solving a categorization problem involving nine categories. These include the seven factuality errors, along with an additional category "other error" for error outside the seven errors, and an additional category "no error" for cases whether no error was detected. Given a pair of input text and model summary, the LLM is expected to output the error type classified into one of the nine categories for each sentence along with a concise reason.
  
- **Keyfact Alignment:** This is an alignment task of matching each keyfact into the summary sentences from which the keyfact is inferable. Given a pari of keyfact list and model summary, the output should be the binary label (whether inferable or not) an dth elist of line numbers of all summary sentences matched for each keyfact.

<p align="center">
<img width="755" alt="스크린샷 2024-07-02 오후 4 56 27" src="https://github.com/DISL-Lab/FineSurE-ACL24/assets/10972556/e5c733b7-d863-4e39-92ac-98f63e8bbee5">
</p>

## FineSurE-Ext: Extensions Completed

This repository extends the original FineSurE pipeline with open-source model support, enhanced factuality reasoning, and analysis utilities for proposal/report-driven experiments.

### 1) Open-source and multi-backend evaluation
- Added open-source fact-checking and keyfact alignment runners:
  - `finesure/fact-checking-opensource.py`
  - `finesure/fact-checking-opensource-enhanced.py`
  - `finesure/keyfact-alignment-opensource.py`
- Added centralized model/backend routing in `finesure/model_config.py` to support multiple model families and serving modes.

### 2) Enhanced factuality pipeline
- Implemented an enhanced two-stage factuality setup (entailment-first + category assignment) in `finesure/fact-checking-opensource-enhanced.py`.
- Added run diagnostics and summaries to improve error analysis and reproducibility:
  - `enhanced-diagnostics.jsonl`
  - `enhanced-summary.json`

### 3) Broader experiment coverage and reproducibility
- Added experiment presets in `finesure/factuality_experiment_presets.py`.
- Expanded reproducible result artifacts for multiple model variants (Qwen, Qwen-3B, Llama, GPT-4) under `reproduce/results/` and `reproduce/results/reports/`.
- Added plotting utility `reproduce/plot_evaluation_reports.py` to generate report-ready visualizations.

### 4) Report-backed observations from current experiments
- Enhanced Qwen (7B) improves faithfulness quality over base Qwen on key headline metrics.
- Enhanced Llama currently underperforms base Llama due to conservative error detection behavior (over-predicting "no error").
- To reduce unnecessary constraints, we removed extra generation restrictions in the enhanced Llama path (for example: forced token cap and rigid optional decoding constraints), while keeping required prompt-format compatibility.

## Proposed Next Steps (Roadmap)

The next phase follows the proposal/report direction and focuses on improving robustness, calibration, and usability:

1. Stage-1 calibration for enhanced Llama
  - Improve not-entailed detection sensitivity to reduce false "no error" predictions.

2. Prompt and reasoning ablations
  - Compare single-stage vs two-stage prompts and controlled reasoning variants across FRANK and REALSumm.

3. Category-aware reliability improvements
  - Add targeted handling for low-recall fine-grained error categories and class imbalance.

4. Efficiency and reliability tracking
  - Add standardized latency/cost/success logging across model backends.

5. Better experiment UX
  - Consolidate runners and plotting into a cleaner CLI workflow with consistent output schemas.

## Running FineSurE on Model Summareis

We create sample datasets with 10 examples for fact-checking and keyfact-alignment tasks, respectively.

Please replace the ```openai api key``` with your api key in ```finesure/fact-checking.py``` and ```finesure/keyfact-alignmnet.py```.

#### Runnining Command:
```bash
cd CodeRelease
python finesure/fact-checking.py [input-path] [output-folder]

# example code for fact checking on sampled data.
python finesure/fact-checking.py dataset/frank/frank-data-sample-10.json result/fact-checking
```

#### Runnining Command:
```bash
cd CodeRelease
python finesure/keyfact-alignment.py [input-path] [keyfact-path] [output-folder]

# example code for keyfact alignment on sampled data.
python finesure/keyfact-alignment.py dataset/realsumm/realsumm-data-sample-10.json dataset/realsumm/human-keyfact-list.json result/keyfact-alignment
```

#### Logs:

The results are saved in the result directory. See the results on examples below:

* Fact Checking Task:
```bash 
[Evaluation Results]
* sentence-level factuality error ratio per model (lower is better)
bert_sum	0.0%
bus	33.3%
pgn	16.7%
s2s	83.3%
bart	33.3%

* summary-level faithfulness score per model (higher is better)
bert_sum	100.0%
bus	66.7%
pgn	83.3%
s2s	16.7%
bart	75.0%

* system-level model ranking (left is better)
['bert_sum', 'pgn', 'bart', 'bus', 's2s']

* success rate: 100.0%
```

* Keyfact Alignment Task:
```bash 
[Evaluation Results]

* completeness score per model (higher is better)
unilm_out_v2	45.5%
t5_out_large	59.0%

* completeness model ranking (left is better)
['t5_out_large', 'unilm_out_v2']

* conciseness score per model (higher is better)
unilm_out_v2	76.0%
t5_out_large	81.7%

* conciseness model ranking (left is better)
['t5_out_large', 'unilm_out_v2']

* success rate: 100.0%
```

## Reproduce the Main Table of the Paper

```bash
cd CodeRelease/reproduce
python reproduce-main-results.py results/frank-result-by-gpt4-w-finesure.json results/realsumm-result-by-gpt4-w-finesure.json
```


## Citation

Please consider citation if our paper is useful in your research.

```BibTeX
@inproceedings{song2024finesure,
  title={FineSurE: Fine-grained Summarization Evaluation using LLMs},
  author={Song, Hwanjun and Su, Hang and Shalyminov, Igor and Cai, Jason and Mansour, Saab},
  booktitle={ACL},
  year={2024}
}
```
