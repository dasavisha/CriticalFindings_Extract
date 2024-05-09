import sys
import argparse
import pandas as pd 
from tqdm import tqdm
import torch 
import torch.nn as nn 
from transformers import AutoTokenizer, AutoModelForCausalLM 
from findings_extract import findings_extract
from keyterm_extract import keyterm_extract


def main():
    ##argument parsing
    parser = argparse.ArgumentParser(description='This is a script for extracting migrane frequency from clinic notes.')

    ##add the arguments
    parser.add_argument("--model_name", "-m", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                    help='the name of the pre-trained model to be loaded')
    parser.add_argument('--device', "-dev",  type=str, default="0",
                    help='device id')
    parser.add_argument("--prompt_type", "-t", type=str, default="ZS",
                    help='the type of prompting of the pre-trained model; choose from few-shot (FS) or zero-shot (ZS). Default ZS')
    parser.add_argument('--dataset_name', "-d", type=str, default="data_file.csv",
                    help='provide a .csv file where the IMPRESSION and FINDINGS columns stores the text as separate two columns.')
    parser.add_argument("--gen_type", "-g", type=str, default="greedy",
                    help='the type of generation algorithm for inference; choose from greedy or beam_search. Default greedy.')
    parser.add_argument('--responses_output_file', "-gr", type=str, required=True, default="generated_responses_MISTRAL_ZS.csv",
                    help='the output file path to save the retrieved findings from LLM')
    parser.add_argument('--criticalterms', "-ct", type=str, default="CriticalFindingsExpanded.txt",
                    help='the file containing the critical terms to be extracted')
    parser.add_argument('--criticalterms_output_file', "-cr", type=str, required=True, default="criticalterms_MISTRAL_ZS.csv",
                    help='the output file path to save the retrieved critical terms by keyword extractor')
    
    ##parse the arguments 
    args = parser.parse_args()
    model_name = args.model_name
    device = "cuda:{}".format(args.device)
    responses_output_file = args.responses_output_file
    prompt_type = args.prompt_type
    gen_type = args.gen_type
    term_file = args.criticalterms
    criticalterms_output_file = args.criticalterms_output_file

    ##read the input dataset
    #input data pre-processing
    file_path = args.dataset_name #path to the dataset CSV
    print ("Loading the data set...{}".format(file_path))
    data_df = pd.read_csv(file_path)
    print ("Data Shape: ", data_df.shape) 


    print ("Running the {} to get the responses. ".format(model_name))
    try:
        responses_df =  findings_extract(data_df, model_name, device, prompt_type, gen_type, responses_output_file)
        print ("Data Shape: ", responses_df.shape)
        print ("Data Columns: ", responses_df.columns)

    except:
        print('Issues in running: exiting!')
        sys.exit(1)


    print ("Running the critical-findings term extractor. Finding term list from {}".format(term_file))
    try:
        keyterm_extract(responses_df, term_file, criticalterms_output_file)

    except:
        print('Issues in running: exiting!')
        sys.exit(1)

if __name__ == '__main__':
    main()
