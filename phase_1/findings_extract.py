import sys
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

def findings_extract(data_df, model_name, device, prompt_type, gen_type, responses_output_file):

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    ##extract the Mistral 7B instruction fine-tuned model
    torch.manual_seed(42) #for reproducibility  
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)

    

    ## get the data reports 
    report_txtlist = data_df['report']
    

    # dict_data = [{'impression': impression, 'finding': finding} for impression, finding in zip(report_txtlist, findings_txtlist)]
    print ("Number of instances loaded ",len(report_txtlist))

    saved_responses_list = list() #a list of all saved responses by the model

    ##the basic prompt with Critical Findings and Incidental Findings definition 
    hyp_base = " The definition of Critical and Incidental findings in a clinical report: CRITICAL findings are life threating imaging findings that needs to be communicated immediately while INCIDENTAL findings non-life-threatening findings, but significant enough that they need to be communicated within a short period of time."

    ##concatenating the above definitions with further information of categories in CRITICAL findings
    hyp_catCF = hyp_base + "CRITICAL findings can be further categorized into three types - NEW, KNOWN/EXPECTED, and UNCERTAIN. The following are the definitions of each of these. NEW: A new critical finding is a critical finding identified in the report to be present for the first time regardless if it is mild/minimal/small in severity, or a critical finding that is implied to have been previously reported but appears to be worsening or increasing in severity. The new critical finding category does not encompass critical findings that are stable, improving, or decreasing in severity. Such critical findings should be classified as known/expected. KNOWN/EXPECTED: A known/expected critical finding is a critical finding that is implied by the report to be known and is described as being either unchanged, improving, or decreasing in severity. This category does not include critical findings that are known but are worsening or increasing in severity; such findings should be categorized as new critical findings. This category also covers critical findings which may be anticipated as part of post-surgical changes. UNCERTAIN: An uncertain critical finding is a critical finding that isn't definitively confirmed in the report. These critical findings are mentioned as possibly being present, mentioned that they should be considered, or concern/suspicion regarding their presence are raised."

    type_examples = ""
    if prompt_type == "FS": #few-shot
        type_examples = input("Enter the Few-Shot Examples: \n ")
    
    ##creating the input prompts
    for report in tqdm(report_txtlist):
        # context = x
        sentence = f"{report}\n{hyp_base}\n"
        if prompt_type == "ZS": #zero-shot
            print ("Prompting the model in Zero-Shot setting ...")
            message = f"{sentence}\nBased on these above definitions, find the CRITICAL findings and INCIDENTAL finding mentioned in the report. Provide a justification for each CRITICAL finding and categorize it into NEW, KNOWN/EXPECTED, and UNCERTAIN."
        
        
        if prompt_type == "FS": #few-shot
            print ("Prompting the model in Few-Shot setting ...")
            message = f"{sentence}\n{type_examples}\nBased on these above definitions and provided few-shot examples, find the CRITICAL findings and INCIDENTAL finding mentioned in the report. Provide a justification for each CRITICAL finding."# and categorize it into NEW, KNOWN/EXPECTED, and UNCERTAIN."

        prompt = f"<s>[INST] {message} [/INST]"

        ##generating the response
        input_ids = tokenizer.encode(prompt+"\n\nANSWER:", return_tensors='pt', return_attention_mask=False).to(device)
        with torch.no_grad(): 
            if gen_type == 'greedy':
                output = model.generate(input_ids, max_new_tokens=512, do_sample=True) ##greedy output
            if gen_type == 'beam_search':
                output = model.generate(input_ids, max_new_tokens=512, num_beams=5, early_stopping=True) ##beam search output with 5 beams 

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        ##preprocess the response to extract only the Model Output
        extr_response = extract_model_response(response)
        tuple_ = (report, extr_response)
        saved_responses_list.append(tuple_)

    print ("Saving the data to a CSV file ...")
    print ("Number of extracted responses: ", len(saved_responses_list))
    generated_responses_df = pd.DataFrame(saved_responses_list, columns =['report', 'response'])

    ##save the list of tuples to a csv file 
    print ("Saving extracted data to a file ...")
    generated_responses_df.to_csv(responses_output_file, index=False)
    return generated_responses_df

if __name__ == "__main__":

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device_id = "0" 
    device = "cuda:{}".format(device_id)
    file_path = "../data/mimic_testcases.csv"
    responses_output_file = "generated_responses_mimic_testdata_MISTRAL_ZS.csv" #num_samples, FS/ZS
    prompt_type = "ZS" #ZS
    gen_type = "greedy"

    print ("Loading the data set...{}".format(file_path))
    data_df = pd.read_csv(file_path)
    print ("Data Shape: ", data_df.shape) 

    ##drop the rows with missing information
    data_df = data_df.dropna()
    print ("Data Shape: ", data_df.shape) 

    print ("Running the {} to get the responses. ".format(model_name))
    responses_df =  findings_extract(data_df, model_name, device, prompt_type, gen_type, responses_output_file)
    print ("Data Shape: ", responses_df.shape)
    print ("Data Columns: ", responses_df.columns)

