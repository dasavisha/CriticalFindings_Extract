import os
import sys
import re
import json
from tqdm import tqdm
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import AutoPeftModelForCausalLM

##CUDA_VISIBLE_DEVICES=3 python inference_mistral.py

##data processing helper functions
def remove_newlinechars(my_string):
    my_string = re.sub(r'\r\n', '', my_string)
    return my_string

def create_text_row(instruction, input, output):
    # this function is used to output the right formate for each row in the dataset
    text_row = f"""<s>[INST] {instruction} Here are the findings and response from the report: {input} [/INST]. And the extracted terms for the critical findings are:  """
    return text_row

def process_input_file(data_df):
    ## get the data impression and findings for the CT data
    impression_txtlist = data_df['impression']
    findings_txtlist = data_df['finding']
    responses_txtlist = data_df['response']
    matched_termslist = data_df['Exact_Match_Terms']

    dict_data = [{'impression': impression, 'finding': finding, 'matched_terms': matched_terms, 'responses': responses} for impression, finding, matched_terms, responses in zip(impression_txtlist, findings_txtlist, matched_termslist, responses_txtlist)]
    print ("Number of instances loaded ",len(dict_data))

    ##the basic prompt with Critical Findings and Incidental Findings definition 
    hyp_base = "The definition of Critical and Incidental findings in a clinical report: CRITICAL findings are life threating imaging findings that needs to be communicated immediately while INCIDENTAL findings non-life-threatening findings, but significant enough that they need to be communicated within a short period of time."

    # ##concatenating the above definitions with further information of categories in CRITICAL findings
    # hyp_catCF = hyp_base + "CRITICAL findings can be further categorized into three types - NEW, KNOWN/EXPECTED, and UNCERTAIN. The following are the definitions of each of these. NEW: A new critical finding is a critical finding identified in the report to be present for the first time regardless if it is mild/minimal/small in severity, or a critical finding that is implied to have been previously reported but appears to be worsening or increasing in severity. The new critical finding category does not encompass critical findings that are stable, improving, or decreasing in severity. Such critical findings should be classified as known/expected. KNOWN/EXPECTED: A known/expected critical finding is a critical finding that is implied by the report to be known and is described as being either unchanged, improving, or decreasing in severity. This category does not include critical findings that are known but are worsening or increasing in severity; such findings should be categorized as new critical findings. This category also covers critical findings which may be anticipated as part of post-surgical changes. UNCERTAIN: An uncertain critical finding is a critical finding that isn't definitively confirmed in the report. These critical findings are mentioned as possibly being present, mentioned that they should be considered, or concern/suspicion regarding their presence are raised."

    instruction = f"{hyp_base} Based on these above definitions, find the CRITICAL FINDINGS TERMS mentioned in the following input report."

    all_eval_prompts = list()
    for x in tqdm(range(len(dict_data))):
        context = remove_newlinechars(dict_data[x]['impression'] + dict_data[x]['finding'])
        prompt = create_text_row(instruction, context, dict_data[x]['matched_terms'].rstrip())
        all_eval_prompts.append(prompt)
    return (all_eval_prompts)

def process_mimic_file(data_df):
    ##get the mimic dataset
    note_txtlist = data_df['text']
    notetype_txtlist = data_df['note_type']

    dict_data = [{'text': text, 'type': note_type} for text, note_type in zip(note_txtlist, notetype_txtlist)]
    print ("Number of instances loaded ",len(dict_data))

    ##the basic prompt with Critical Findings and Incidental Findings definition 
    hyp_base = "The definition of Critical and Incidental findings in a clinical report: CRITICAL findings are life threating imaging findings that needs to be communicated immediately while INCIDENTAL findings non-life-threatening findings, but significant enough that they need to be communicated within a short period of time."

    # ##concatenating the above definitions with further information of categories in CRITICAL findings
    # hyp_catCF = hyp_base + "CRITICAL findings can be further categorized into three types - NEW, KNOWN/EXPECTED, and UNCERTAIN. The following are the definitions of each of these. NEW: A new critical finding is a critical finding identified in the report to be present for the first time regardless if it is mild/minimal/small in severity, or a critical finding that is implied to have been previously reported but appears to be worsening or increasing in severity. The new critical finding category does not encompass critical findings that are stable, improving, or decreasing in severity. Such critical findings should be classified as known/expected. KNOWN/EXPECTED: A known/expected critical finding is a critical finding that is implied by the report to be known and is described as being either unchanged, improving, or decreasing in severity. This category does not include critical findings that are known but are worsening or increasing in severity; such findings should be categorized as new critical findings. This category also covers critical findings which may be anticipated as part of post-surgical changes. UNCERTAIN: An uncertain critical finding is a critical finding that isn't definitively confirmed in the report. These critical findings are mentioned as possibly being present, mentioned that they should be considered, or concern/suspicion regarding their presence are raised."

    instruction = f"{hyp_base} Based on these above definitions, find the CRITICAL FINDINGS TERMS mentioned in the following input report."

    all_eval_prompts = list()
    for x in tqdm(range(len(dict_data))):
        context = remove_newlinechars(dict_data[x]['text'] + dict_data[x]['type'])
        prompt = create_text_row(instruction, context, " ")
        all_eval_prompts.append(prompt)
    return (all_eval_prompts[:3000])


def main():
    ##load a pretrained model 
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print ("loading the fine-tuned model from path ...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # path_to_ft_model = "./results_15K/checkpoint-1000/" #mistral model fine-tuned with ZERO SHOT Phase 1 examples
    path_to_ft_model = "./results_15K_FS/checkpoint-1000/" #mistral model fine-tuned with FEW SHOT Phase 1 examples
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    # base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(path_to_ft_model, torch_dtype=torch.float16) 
    # model = AutoPeftModelForCausalLM.from_pretrained(path_to_ft_model, load_in_4bit=True)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    ##generating the responses from the model 
    # test_data_file = "./outputs/phase_1/extracted_terms/ZS/criticalterms_MISTRAL_ZS_chest_testdataV2.csv"
    test_data_file = "../Data_Avisha/inputs/radiology.csv" ##for mimic-iv-notes
    print ("Loading the data set...{}".format(test_data_file))
    data_df = pd.read_csv(test_data_file)
    print ("Data Shape: ", data_df.shape) 
    print ("Data Columns: ", data_df.columns)
    # all_eval_prompts = process_input_file(data_df) ##for the chest data
    all_eval_prompts = process_mimic_file(data_df) ##for mimic-iv-notes (2321355)
    # print (remove_newlinechars(all_eval_prompts[0]))
    print ("Num instances:", len(all_eval_prompts)) 
    

    print ("starting the inference.")
    all_test_responses = list()
    model.eval()
    for eval_prompt in tqdm(all_eval_prompts):
        model_input = tokenizer(eval_prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            generated_code = tokenizer.decode(model.generate(model_input, max_new_tokens=100, pad_token_id=2)[0], skip_special_tokens=True)
        response_extracted = generated_code.replace(eval_prompt, '')      
        tuple_response = (eval_prompt, response_extracted)
        all_test_responses.append(tuple_response)
    
    ##save to file
    df = pd.DataFrame(all_test_responses, columns =['input_report', 'Extracted_Terms'])
    print ("writing to csv file.")
    # outfile = "weakoutput_MISTRAL_15KFS_chesttestdata.csv"
    outfile = "weakoutput_MISTRAL_15KFS_mimic_iv_radiology_3K.csv"
    df.to_csv(outfile, index=False)  

    ##scrap: 
    # eval_prompt = """Print hello world in python c and c++"""
    # model_input = tokenizer(eval_prompt, return_tensors="pt").input_ids.to(device)
    
    # with torch.no_grad():
    #     generated_code = tokenizer.decode(model.generate(model_input, max_new_tokens=100, pad_token_id=2)[0], skip_special_tokens=True)
    # print(generated_code)

if __name__ == "__main__":
    main()

