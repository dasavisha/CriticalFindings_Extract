import sys 
import os 
import pandas as pd 
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from statistics import mean, stdev
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

##this is for running evaluation for prometheus on the datasets
## CUDA_VISIBLE_DEVICES=2 python LLM_eval_prom.py
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
access_token =  ""
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True, token=access_token)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
model_prometheus = LlamaForCausalLM.from_pretrained("kaist-ai/Prometheus-13b-v1.0", device_map="auto", torch_dtype=torch.float16)
model_prometheus.to(device)
    


#input CSV file 
print ("processing the data...")
csv_input = "./eval_outputs/all_models_MIMIC.csv"
data_df = pd.read_csv(csv_input)
print ("Columns:", data_df.columns)

all_models = (list(data_df.columns))[3:]  
print (all_models)

##for MIMIC data
input_reports = data_df['report'].tolist()
data_df['critical_finding'] = data_df['critical_finding'].fillna(' ')
ground_truth_answers = data_df['critical_finding'].tolist()
data_df['critical_GPT4'] = data_df['critical_GPT4'].fillna('[ ]')
data_df['critical_BioMISTRAL_7B_FT_FS'] = data_df['critical_GPT4'].fillna('[ ]')


##running Prometheus on datasets
# print ("Running Prometheus on all models ...")
##creating the Prometheus prompt
task_desc = "###Task Description: An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given. 1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general. 2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric. 3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 3)\" 4. Please do not generate any other opening, closing, and explanations."
instruction_to_eval = "###The instruction to evaluate: "
response_to_eval = "###Response to evaluate: "
reference_answer = "###Reference Answer (Score 3): "
score_rubrics = "###Score Rubrics: [Correctness - determine if the actual output is correct according to the expected output. Look for partial matches.] Score 1: The model completely fails to generate the output. Score 2: The model generates some part of the reference output. Score 3: The model exactly identifies and generates the reference answer."


# ##testing with an example
# item = "1 . Multifocal bilateral pneumonia with right lung cavitary lesions , right calcified granulomas , and right pleural plaques are very concerning for reactivation tuberculosis with a component of right upper lobe necrotizing pneumonia . 2 . Enlarged pulmonary artery suggesting underlying pulmonary hypertension . 3 . Hard and soft plaque throughout the aorta with narrowing of the origin of the celiac artery . Finding # 1 was discussed with Dr . First Name8 ( NamePattern2 ) Last Name ( NamePattern1 ) 92986 by phone at 3 : 40 p . m . on 2192-9 -11 immediately after discovery and attending review ."
# model_answer = "['active tubercolosis']"
# ground_truth_answer = "reactivation tuberculosis necrotizing pneumonia"

# input_str = item + " The definition of Critical findings in a clinical report: CRITICAL findings are life threatening imaging findings that needs to be communicated immediately. Get the CRITICAL_FINDINGS from the report."
# instruction_to_eval = instruction_to_eval + input_str
# response_to_eval = response_to_eval + model_answer
# reference_answer = reference_answer + ground_truth_answer
# prompt_prometheus = task_desc + instruction_to_eval + response_to_eval + reference_answer + score_rubrics + "###Feedback: "
# input_ids = tokenizer(prompt_prometheus, return_tensors="pt").input_ids.to(device)
# outputs = model.generate(input_ids, max_new_tokens=512)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

##evaluating the models
models = ['critical_GPT4', 'critical_MISTRAL_7B_ZS', 'critical_MISTRAL_7B_FS', 'critical_BioMISTRAL_7B_ZS', 'critical_BioMISTRAL_7B_FS', 'critical_MISTRAL_7B_FT_ZS', 'critical_MISTRAL_7B_FT_FS', 'critical_BioMISTRAL_7B_FT_ZS', 'critical_BioMISTRAL_7B_FT_FS']
print ("Running Prometheus on model one by one ...")

model_num = int(sys.argv[1]) ## enter the model number 
start_idx = int(sys.argv[2]) ## enter the start index
end_idx = int(sys.argv[3]) ## enter the end index
name_suffix = sys.argv[4] ## enter the suffix 

model = models[model_num] ## change this 0 to 7
model_answers =  data_df[model].tolist()
print (model_answers)
prometheus_outputs_model = list()
for i, item in enumerate(input_reports[start_idx:end_idx]):  ## change this :20, 20:40, 40:
    input_str = item + " The definition of Critical findings in a clinical report: CRITICAL findings are life threatening imaging findings that needs to be communicated immediately. Get the CRITICAL_FINDINGS from the report."
    instruction_to_eval = instruction_to_eval + input_str
    # print (model_answers[i])
    response_to_eval = response_to_eval + model_answers[i]
    # print (ground_truth_answers[i])
    reference_answer = reference_answer + ground_truth_answers[i]
    prompt_prometheus = task_desc + instruction_to_eval + response_to_eval + reference_answer + score_rubrics + "###Feedback: "
    input_ids = tokenizer(prompt_prometheus, return_tensors="pt", padding="max_length", max_length=100).input_ids.to(device)
    outputs = model_prometheus.generate(input_ids, max_new_tokens=100)
    decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    extracted_feedback = decoded_outputs.replace(prompt_prometheus, "Feedback")
    tuple_ = (item, model_answers[i], ground_truth_answers[i], extracted_feedback)
    # print (i, item, extracted_feedback)
    print (i)
    prometheus_outputs_model.append(tuple_)
    # print(tokenizer.decode(outputs[0]), skip_special_tokens=True)
    # dict_allmodels_prometheus[model].append(extracted_feedback)

##saving the scores and reasons in a file
print ("Saving the Prometheus results to a file.")
filename = os.path.join("./PrometheusOutputs-001a/", "Prometheus_Model_"+model+name_suffix) #top20, 20_40, 40_60, 60_80, 80_100, 100_
data_df_geval = pd.DataFrame(prometheus_outputs_model, columns =['Report', 'Model_Answer', 'GroundTruth_Answer', 'Score'])
data_df_geval.to_csv(filename)     
