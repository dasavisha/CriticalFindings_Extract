#!/bin/bash
cd phase_1
echo "running the Phase 1 model on MIMIC Test Cases"

## running the model in ZS mode
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "mistralai/Mistral-7B-Instruct-v0.2" --prompt_type "ZS" --dataset_name "../data/mimic_testcases.csv" --responses_output_file "./generated_responses_mimic_testdata_MISTRAL_ZS.csv" --criticalterms_output_file "./criticalterms_mimic_testdata_MISTRAL_ZS.csv"

## running the model in FS mode
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "mistralai/Mistral-7B-Instruct-v0.2" --prompt_type "FS" --dataset_name "../data/mimic_testcases.csv" --responses_output_file "./generated_responses_mimic_testdata_MISTRAL_FS.csv" --criticalterms_output_file "./criticalterms_mimic_testdata_MISTRAL_FS.csv"