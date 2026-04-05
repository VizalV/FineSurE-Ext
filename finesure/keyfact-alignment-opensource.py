"""
Keyfact alignment evaluation with open-source model support
"""
import json
import sys
import os
from utils_opensource import (
    get_response_opensource,
    get_keyfact_alighment_prompt, 
    parsing_llm_keyfact_alighment_output,
    compute_completeness_percentage_score, 
    compute_conciseness_percentage_score
)

def main(input_path, keyfact_path, output_path, model_key="qwen2.5-7b", print_interval=2):
    '''
    Argument:
        input_path: path for input data
        keyfact_path: path for human or machine keyfacts
        output_path: path for output data (saving the logs and the eval results)
        model_key: model to use (from model_config.py)
        print_interval: print the percentage scores every 'print_interval' 
    '''

    print(f"Using model: {model_key}")
    print(f"Input: {input_path}")
    print(f"Keyfacts: {keyfact_path}")
    print(f"Output: {output_path}\n")

    # loads data for completeness and conciseness evaluation using FineSurE
    inputs = []
    for line in open(input_path, 'r'):
        line = json.loads(line)
        inputs.append(line)

    # loads keyfacts 
    keyfacts = {}
    for line in open(keyfact_path, 'r'):
        line = json.loads(line)
        keyfacts[line['doc_id']] = line['key_facts']

    # variables for evaluation 
    cnt_total_inference = 0
    cnt_success_inference = 0
    model_labels = {}
    
    # writer to store the output from LLM evaluation
    os.makedirs(output_path, exist_ok=True)
    raw_data_writer = open(os.path.join(output_path, 'raw-data.json'), 'w')
    result_writer = open(os.path.join(output_path, 'result.json'), 'w')

    # processes each data instance using for loop
    for input_id, input_json in enumerate(inputs):

        # input json parsing
        doc_id = input_json['doc_id']
        model_name = input_json['model']
        src = input_json['transcript']
        sentences = input_json['sentences']
        list_keyfacts = keyfacts[doc_id]

        # prompt generation
        prompt = get_keyfact_alighment_prompt(keyfacts=list_keyfacts, sentences=sentences)

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
            input_json['pred_alignment_labels'], input_json['pred_sentence_line_numbers'] = parsing_llm_keyfact_alighment_output(output)

            # check if the parsing is success
            success_flag = True
            if len(input_json['pred_alignment_labels']) == 0:
                success_flag = False

            print(f"Success: {success_flag}")
            print(f'\t[Alignment Label]: {input_json["pred_alignment_labels"]}')
            print(f'\t[Matched Sentence Line Numbers]: {input_json["pred_sentence_line_numbers"]}')

            # count the success cases
            cnt_total_inference += 1
            if success_flag:
                cnt_success_inference += 1
            else:
                # fail to evaluate -> skip
                print("\t⚠️  Skipping - parsing failed")
                continue      

            # compute the percentage score for completeness and conciseness
            completeness_score = compute_completeness_percentage_score(input_json['pred_alignment_labels'])
            conciseness_score = compute_conciseness_percentage_score(input_json['pred_sentence_line_numbers'], len(sentences))

            # put the score into the aggregation dictionary
            if model_name not in model_labels:
                model_labels[model_name] = {'completeness_scores': [], 'conciseness_scores': []}
            model_labels[model_name]['completeness_scores'].append(completeness_score)
            model_labels[model_name]['conciseness_scores'].append(conciseness_score)

            print(f'\t[Completeness Score]: {completeness_score:.1%}')
            print(f'\t[Conciseness Score]: {conciseness_score:.1%}')

        except Exception as e:
            print(f"\t❌ Error processing example: {e}")
            import traceback
            traceback.print_exc()
            continue

        def print_results_completeness(model_labels):
            summary_level_completeness_scores = {}
            summary_level_conciseness_scores = {}

            for model_name, error_labels in model_labels.items():
                summary_level_completeness_scores[model_name] = sum(error_labels['completeness_scores']) / len(error_labels['completeness_scores'])
                summary_level_conciseness_scores[model_name] = sum(error_labels['conciseness_scores']) / len(error_labels['conciseness_scores'])

            text_output = "\n\n\n[Evaluation Results]\n"
            text_output += '\n* completeness score per model (higher is better)\n'
            for model_name, score in summary_level_completeness_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            text_output += '\n* completeness model ranking (left is better)\n'
            sorted_dict = dict(sorted(summary_level_completeness_scores.items(), key=lambda item: item[1], reverse=True))
            model_ranking = list(sorted_dict.keys())
            text_output += str(model_ranking) + '\n'

            text_output += '\n* conciseness score per model (higher is better)\n'
            for model_name, score in summary_level_conciseness_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            text_output += '\n* conciseness model ranking (left is better)\n'
            sorted_dict = dict(sorted(summary_level_conciseness_scores.items(), key=lambda item: item[1], reverse=True))
            model_ranking = list(sorted_dict.keys())
            text_output += str(model_ranking) + '\n'

            success_ratio = '{:.1%}'.format(cnt_success_inference/float(cnt_total_inference))
            text_output += '\n* success rate: ' + str(success_ratio) + '\n\n\n'

            print(text_output)
            return text_output

        # print percentage score
        if cnt_total_inference % print_interval == 0:
            print_results_completeness(model_labels=model_labels)
           
        json.dump(input_json, raw_data_writer)
        raw_data_writer.write('\n')
        raw_data_writer.flush()
        
    raw_data_writer.close()

    # print final results
    if model_labels:
        text_output = print_results_completeness(model_labels=model_labels)
        result_writer.write(text_output)
    else:
        print("\n⚠️  No successful evaluations!")
        
    result_writer.close()
    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    '''
    Running Command:
        python finesure/keyfact-alignment-opensource.py [input-path] [keyfact-path] [output-folder] [model-key]
        
        e.g., python finesure/keyfact-alignment-opensource.py dataset/realsumm/realsumm-data-sample-10.json dataset/realsumm/human-keyfact-list.json result/keyfact-qwen qwen2.5-7b
    '''

    if len(sys.argv) < 4:
        print("Usage: python keyfact-alignment-opensource.py <input_path> <keyfact_path> <output_path> [model_key]")
        print("Available models: qwen2.5-7b, qwen3, deepseek-r1, qwen2.5-3b, qwen2.5-7b-direct")
        sys.exit(1)

    input_path = sys.argv[1]
    keyfact_path = sys.argv[2]
    output_path = sys.argv[3]
    model_key = sys.argv[4] if len(sys.argv) > 4 else "qwen2.5-7b"

    # print logs every 10 inferences
    print_interval = 10

    main(input_path, keyfact_path, output_path, model_key, print_interval)
