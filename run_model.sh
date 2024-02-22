python main.py \
--model_name "mistralai/Mistral-7B-Instruct-v0.2" \
--dataset_name "./data_file.csv" \
--output_file "./generated_responses_MISTRAL_ZS.csv" \
--type "ZS"

python keyterm_extract.py generated_responses_MISTRAL_ZS.csv extracted_keyterms_MISTRAL_ZS.csv
