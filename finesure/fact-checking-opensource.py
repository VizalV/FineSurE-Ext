"""
Fact-checking evaluation with open-source model support
"""
import json
import sys
import os
from utils_opensource import (
    get_response_opensource, 
    parsing_llm_fact_checking_output, 
    compute_faithfulness_percentage_score,
    get_fact_checking_prompt
)

def main(input_path, output_path, model_key="qwen2.5-7b", print_interval=2):
    '''
    Argument:
        input_path: path for input data
        output_path: path for output data (saving the logs and the eval results)
        model_key: model to use (from model_config.py)
        print_interval: print the percentage scores every 'print_interval' 
    '''

    print(f"Using model: {model_key}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}\n")

    # loads data for faithfulness evaluation using FineSurE
    inputs = []
    for line in open(input_path, 'r'):
        line = json.loads(line)
        inputs.append(line)

    # variables for evaluation 
    cnt_total_inference = 0
    cnt_success_inference = 0
    model_labels = {}
    
    # writer to store the output from LLM evaluation
    os.makedirs(output_path, exist_ok=True)
    raw_data_writer = open(os.path.join(output_path, 'raw-data.json'), 'w')
    result_writer = open(os.path.join(output_path, 'result.json'), 'w')
    llama_failure_writer = open(os.path.join(output_path, 'llama-parse-failures.jsonl'), 'w')

    # processes each data instance using for loop
    for input_id, input_json in enumerate(inputs):

        # input json parsing
        doc_id = input_json['doc_id']
        model_name = input_json['model']
        src = input_json['transcript']
        sentences = input_json['sentences']

        # prompt generation
        prompt = get_fact_checking_prompt(input=src, sentences=sentences)

        print(f"\n{'='*80}")
        print(f"[{input_id+1}/{len(inputs)}] Processing doc_id: {doc_id}, model: {model_name}")
        print(f"{'='*80}")

        try:
            # get response from LLM
            output = get_response_opensource(
                prompt=prompt, 
                model_key=model_key,
                temperature=0.0,
                max_tokens=2048
            )
     
            input_json['llm_output'] = output
            input_json['pred_faithfulness_labels'], input_json['pred_faithfulness_error_type'] = parsing_llm_fact_checking_output(output)

            # check if the parsing is success
            success_flag = True
            if len(input_json['pred_faithfulness_labels']) == 0:
                success_flag = False
            # Llama completions must return one label per summary sentence.
            if len(input_json['pred_faithfulness_labels']) != len(sentences):
                if "llama" in model_key.lower():
                    success_flag = False
                    llama_failure_writer.write(json.dumps({
                        "doc_id": doc_id,
                        "model": model_name,
                        "num_sentences": len(sentences),
                        "parsed_label_count": len(input_json['pred_faithfulness_labels']),
                        "output_preview": output[:2000],
                    }) + "\n")
                    llama_failure_writer.flush()
                    input_json['pred_faithfulness_labels'] = []
                    input_json['pred_faithfulness_error_type'] = []

            print(f"Success: {success_flag}")
            print(f'\t[Error Label]: {input_json["pred_faithfulness_labels"]}')
            print(f'\t[Error Type]: {input_json["pred_faithfulness_error_type"]}')

            # count the success cases
            cnt_total_inference += 1
            if success_flag:
                cnt_success_inference += 1
            else:
                # fail to evaluate -> skip
                print("\t⚠️  Skipping - parsing failed")
                continue

            # compute the percentage score for faithfulness
            faithfulness_score = compute_faithfulness_percentage_score(input_json['pred_faithfulness_labels'])
            print(f'\t[Faithfulness Score]: {faithfulness_score:.1%}')

            # put the score into the aggregation dictionary
            if model_name not in model_labels:
                model_labels[model_name] = {'faithfulness_scores': [], 'binary_labels': []}
            model_labels[model_name]['faithfulness_scores'].append(faithfulness_score)
            model_labels[model_name]['binary_labels'].extend(input_json['pred_faithfulness_labels'])

        except Exception as e:
            print(f"\t❌ Error processing example: {e}")
            import traceback
            traceback.print_exc()
            continue

        def print_results_faithfulness(model_labels):
            sentence_level_errors = {}
            summary_level_scores = {}
            for model_name, error_labels in model_labels.items():
                sentence_level_errors[model_name] = sum(error_labels['binary_labels']) / len(error_labels['binary_labels'])
                summary_level_scores[model_name] = sum(error_labels['faithfulness_scores']) / len(error_labels['faithfulness_scores'])

            text_output = "\n\n\n[Evaluation Results]\n"
            text_output += '* sentence-level factuality error ratio per model (lower is better)\n'
            for model_name, error_rate in sentence_level_errors.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(error_rate)) + '\n'

            text_output += '\n* summary-level faithfulness score per model (higher is better)\n'
            for model_name, score in summary_level_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            text_output += '\n* system-level model ranking (left is better)\n'
            sorted_dict = dict(sorted(summary_level_scores.items(), key=lambda item: item[1], reverse=True))
            model_ranking = list(sorted_dict.keys())
            text_output += str(model_ranking) + '\n'

            success_ratio = '{:.1%}'.format(cnt_success_inference/float(cnt_total_inference))
            text_output += '\n* success rate: ' + str(success_ratio) + '\n\n\n'

            print(text_output)
            return text_output

        # print percentage score
        if cnt_total_inference % print_interval == 0:
            print_results_faithfulness(model_labels=model_labels)
           
        json.dump(input_json, raw_data_writer)
        raw_data_writer.write('\n')
        raw_data_writer.flush()
        
    raw_data_writer.close()
    llama_failure_writer.close()

    # print final results
    if model_labels:
        text_output = print_results_faithfulness(model_labels=model_labels)
        result_writer.write(text_output)
    else:
        print("\n⚠️  No successful evaluations!")
        
    result_writer.close()
    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    '''
    Running Command:
        python finesure/fact-checking-opensource.py [input-path] [output-folder] [model-key]
        
        e.g., python finesure/fact-checking-opensource.py dataset/frank/frank-data-sample-10.json result/fact-checking-qwen qwen2.5-7b
    '''

    if len(sys.argv) < 3:
        print("Usage: python fact-checking-opensource.py <input_path> <output_path> [model_key]")
        print("Available models: qwen2.5-7b, qwen3, deepseek-r1, qwen2.5-3b, qwen2.5-7b-direct")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_key = sys.argv[3] if len(sys.argv) > 3 else "qwen2.5-7b"

    # print logs every 10 inferences
    print_interval = 10

    main(input_path, output_path, model_key, print_interval)
