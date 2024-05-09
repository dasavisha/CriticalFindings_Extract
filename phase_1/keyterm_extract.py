import sys
import pandas as pd 
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from fuzzywuzzy import fuzz


def exact_term_extraction(critical_findings_refterms, answer_text):
    ##Type 1: Exact matching - look  if a crtitical reference term is present in generated responses. 
    response_val = list()
    for query_str in critical_findings_refterms:
        query_str = query_str.strip()
        if query_str.lower() in answer_text.lower():
            response_val.append(query_str)
    return response_val

def partial_term_extraction(critical_findings_refterms, answer_text):
    ##Type 2: Partial matching - look  if a crtitical reference term is present partially in generated responses. 
    response_val = list()
    for query_str in critical_findings_refterms:
        query_str = query_str.strip()
        ratio = fuzz.partial_ratio(query_str.lower(), answer_text.lower())
        if ratio > 70: #if there is a 75% match 
            response_val.append(query_str)
    return response_val

def keyterm_extract(data_df, reference_file, output_filename):
    print ("Loading the responses from the model ...")
    print ("Shape: ", data_df.shape)
    print ("Loading the reference terms... {}".format(reference_file))
    with open(reference_file, 'r') as rf:
        critical_findings_refterms = rf.readlines()
    print ("Total number of terms ... ", len(critical_findings_refterms))

    print ("Starting the term matching ...\n")
    # data_df['response'] = data_df['impression'] + data_df['findings'] ##**far code for 10000 or 3500 instances 

    answer_texts = data_df['response'].tolist()
    print ("Number of instances to look at:", len(answer_texts))

    exact_match_list = list()
    partial_match_list = list()
    for item in tqdm(answer_texts):
        exact_match = exact_term_extraction(critical_findings_refterms, item)
        exact_match_list.append(exact_match)

        partial_match = partial_term_extraction(critical_findings_refterms, item)
        partial_match_list.append(partial_match)
    

    data_df['Exact_Match_Terms'] = exact_match_list ##**far code
    data_df['Partial_Match_Terms'] = partial_match_list ##**far code

    # ##**far code
    # partial_df = pd.DataFrame(
    # {'Inputs': answer_texts[5000:15000],
    #  'Exact_Match_Terms': exact_match_list,
    #  'Partial_Match_Terms': partial_match_list
    # })


    ##save the list of tuples to a csv file 
    print ("Saving extracted data to a file ...")
    data_df.to_csv(output_filename, index=False)
    # partial_df.to_csv(output_filename, index=False)

if __name__ == "__main__":

    # file_path = "./generated_responses_chest_testdata_MISTRAL_ZS.csv" #**far
    # file_path = "./generated_responses_reports_ab_hd_nk_ch_5K_MISTRAL_ZS.csv"
    file_path = "./generated_responses_reports_ab_hd_nk_ch_15K_MISTRAL_FS.csv" 
    term_file = "CriticalFindingsExpanded.txt"
    criticalterms_output_file = "criticalterms_MISTRAL_reports_ab_hd_nk_ch_15K_MISTRAL_FS.csv" #**far
    print ("Loading the data set...{}".format(file_path))
    data_df = pd.read_csv(file_path)
    print ("Data Shape: ", data_df.shape) 

    ##drop the rows with missing information
    data_df = data_df.dropna()
    print ("Data Shape: ", data_df.shape) 

    keyterm_extract(data_df, term_file, criticalterms_output_file)
