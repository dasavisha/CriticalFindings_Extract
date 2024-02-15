import os
import sys
import argparse
import pandas as pd 
from tqdm import tqdm
import torch 
import torch.nn as nn 
from transformers import AutoTokenizer, AutoModelForCausalLM 

def extract_model_response(response):
    """Extract the text from the generated model response"""
    idx2 = response.index("ANSWER:")
    answer_txt = ''
    for idx_ in range(idx2 + len("ANSWER:") + 1, len(response)):
        answer_txt = answer_txt + response[idx_]
    return answer_txt

def main():
    ##argument parsing
    parser = argparse.ArgumentParser(description='This is a script for extracting migrane frequency from clinic notes.')

    ##add the arguments
    parser.add_argument("--model_name", "-m", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                    help='the name of the pre-trained model to be loaded')
    parser.add_argument("--type", "-t", type=str, default="ZS",
                    help='the type of prompting of the pre-trained model; choose from few-shot (FS) or zero-shot (ZS). Default ZS')
    parser.add_argument('--dataset_name', "-d", type=str, default="data_file.csv",
                    help='provide a .csv file where the IMPRESSION and FINDINGS columns stores the text as separate two columns.')
    parser.add_argument('--output_file', "-o", type=str, required=True, default="generated_responses_MISTRAL_ZS.csv",
                    help='the output file path to save the retrieved findings from LLM')
    
    args = parser.parse_args()

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    device = 'cuda'


    ##extract the Mistral 7B instruction fine-tuned model
    torch.manual_seed(42) #for reproducibility

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model.to(device)

    #input data pre-processing
    file_path = args.dataset_name #path to the dataset CSV
    print ("Loading the data set...{}".format(file_path))
    data_df = pd.read_csv(file_path)
    print ("Data Shape: ", data_df.shape)

    ## get the data impression and findings 
    impression_txtlist = data_df['IMPRESSION']
    findings_txtlist = data_df['FINDINGS']

    dict_data = [{'impression': impression, 'finding': finding} for impression, finding in zip(impression_txtlist, findings_txtlist)]
    print ("Number of instances loaded ",len(dict_data))

    saved_responses_list = list() #a list of all saved responses by the model

    ##the basic prompt with Critical Findings and Incidental Findings definition 
    hyp_base = " The definition of Critical and Incidental findings in a clinical report: CRITICAL findings are life threating imaging findings that needs to be communicated immediately while INCIDENTAL findings non-life-threatening findings, but significant enough that they need to be communicated within a short period of time."

    ##concatenating the above definitions with further information of categories in CRITICAL findings
    hyp_catCF = hyp_base + "CRITICAL findings can be further categorized into three types - NEW, KNOWN/EXPECTED, and UNCERTAIN. The following are the definitions of each of these. NEW: A new critical finding is a critical finding identified in the report to be present for the first time regardless if it is mild/minimal/small in severity, or a critical finding that is implied to have been previously reported but appears to be worsening or increasing in severity. The new critical finding category does not encompass critical findings that are stable, improving, or decreasing in severity. Such critical findings should be classified as known/expected. KNOWN/EXPECTED: A known/expected critical finding is a critical finding that is implied by the report to be known and is described as being either unchanged, improving, or decreasing in severity. This category does not include critical findings that are known but are worsening or increasing in severity; such findings should be categorized as new critical findings. This category also covers critical findings which may be anticipated as part of post-surgical changes. UNCERTAIN: An uncertain critical finding is a critical finding that isn't definitively confirmed in the report. These critical findings are mentioned as possibly being present, mentioned that they should be considered, or concern/suspicion regarding their presence are raised."

    type_examples = ""
    
    ##creating the input prompts
    for x in tqdm(len(dict_data)):
        context = dict_data[x]['impression'] + dict_data[x]['finding']
        sentence = f"{context}\n{hyp_catCF}\n"
        if args.type == "ZS": #zero-shot
            print ("Prompting the model in Zero-Shot setting ...")
            message = f"{sentence}\nBased on these above definitions, find the CRITICAL findings and INCIDENTAL finding mentioned in the report. Provide a justification for each CRITICAL finding and categorize it into NEW, KNOWN/EXPECTED, and UNCERTAIN."
        
        if args.type == "FS": #few-shot
            type_examples = raw_input("Enter the Few-Shot Examples: \n ")

            print ("Prompting the model in Few-Shot setting ...")
            message = f"{sentence}\n{type_examples}\nBased on these above definitions and provided few-shot examples, find the CRITICAL findings and INCIDENTAL finding mentioned in the report. Provide a justification for each CRITICAL finding and categorize it into NEW, KNOWN/EXPECTED, and UNCERTAIN."

        prompt = f"<s>[INST] {message} [/INST]"

        ##generating the response
        input_ids = tokenizer.encode(prompt+"\n\nANSWER:", return_tensors='pt', return_attention_mask=False).to(device)
        with torch.no_grad(): 
            greedy_output = model.generate(input_ids, max_new_tokens=512, do_sample=True) ##greedy output

        response = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        ##preprocess the response to extract only the Model Output
        extr_response = extract_model_response(response)
        tuple_ = (dict_data[x]['impression'], dict_data[x]['finding'], extr_response)
        saved_responses_list.append(tuple_)

    print ("Saving the data to a CSV file ...")
    print ("Number of extracted responses: ", len(saved_responses_list))
    generated_responses_df = pd.DataFrame(saved_responses_list, columns =['impression', 'finding', 'response'])

    ##save the list of tuples to a csv file 
    print ("Saving extracted data to a file ...")
    output_filename = args.output_file
    generated_responses_df.to_csv(output_filename, index=False)

    if __name__ == "__main__":
        main()


