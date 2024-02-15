import os 
import sys
import pandas as pd 
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import time
from pathlib import Path

#python keyterm_extract.py <inputfile> <outputfile>

def term_extraction(critical_findings_refterms, answer_text):
    
    ##Type 1: Exact matching - look  if a crtitical reference term is present in generated responses. 
    response_val = list()

    # print ("Results of Exact Matching ...")
    for query_str in critical_findings_refterms:
        
        query_str = query_str.strip()
        if query_str.lower() in answer_text.lower():
            response_val.append(query_str)
    return response_val


file_path = sys.argv[1] #the extracted incidental and critical findings 
print ("Loading the generated data set...{}".format(file_path))
data_df = pd.read_csv(file_path)
# print (data_df.head(20))
print ("Shape: ", data_df.shape)

reference_file = "CriticalFindings.txt"
print ("Loading the reference terms... {}".format(reference_file))
with open(reference_file, 'r') as rf:
    critical_findings_refterms = rf.readlines()


print ("Starting the term matching ...\n")

answer_texts = data_df['response'].tolist()
print ("Number of instances:", len(answer_texts))

responses_list = list()
for item in tqdm(answer_texts):
    response = term_extraction(critical_findings_refterms, item)
    responses_list.append(response)

data_df['CriticalFindingsTerms'] = responses_list

##save the list of tuples to a csv file 
print ("Saving extracted data to a file ...")
output_filename = sys.argv[2]
data_df.to_csv(output_filename, index=False)