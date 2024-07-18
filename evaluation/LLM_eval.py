#export OPENAI_API_KEY="sk-proj-YqFWbKwA1Jve7yX18RuQT3BlbkFJRcRoHusBUa6e4OoARfut"
import sys 
import os 
import pandas as pd 
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from statistics import mean, stdev
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

##initialize the model
# CUDA_VISIBLE_DEVICES=2 python LLM_eval.py
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
access_token =  ""
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token)
model = LlamaForCausalLM.from_pretrained("kaist-ai/Prometheus-13b-v1.0", device_map="auto", torch_dtype=torch.float16)
model.to(device)

##input CSV file 
# print ("processing the data...")
# csv_input = "./eval_outputs/all_models_MIMIC.csv"
# data_df = pd.read_csv(csv_input)
# print ("Columns:", data_df.columns)

# all_models = (list(data_df.columns))[3:]  
# print (all_models)

# ##for MIMIC data
# input_reports = data_df['report'].tolist()
# ground_truth_answers = data_df['critical_finding'].tolist()



##calculating G-EVAL metric 
##ref: https://github.com/confident-ai/deepeval/blob/main/examples/getting_started/test_example.py
##ref: https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation#g-eval

# def get_geval_metric(test_case):
#     correctness_metric = GEval(
#     name="Correctness",
#     criteria="Correctness - determine if the actual output is correct according to the expected output. Look for partial matches. ",
#     evaluation_params=[
#             LLMTestCaseParams.ACTUAL_OUTPUT,
#             LLMTestCaseParams.EXPECTED_OUTPUT,
#         ])

#     correctness_metric.measure(test_case)
#     return (correctness_metric)
#     # print(correctness_metric.score)
#     # print(correctness_metric.reason)

# ##running G-EVAL on datasets
# test_case = LLMTestCase(input= "1 . Multifocal bilateral pneumonia with right lung cavitary lesions , right calcified granulomas , and right pleural plaques are very concerning for reactivation tuberculosis with a component of right upper lobe necrotizing pneumonia . 2 . Enlarged pulmonary artery suggesting underlying pulmonary hypertension . 3 . Hard and soft plaque throughout the aorta with narrowing of the origin of the celiac artery . Finding # 1 was discussed with Dr . First Name8 ( NamePattern2 ) Last Name ( NamePattern1 ) 92986 by phone at 3 : 40 p . m . on 2192-9 -11 immediately after discovery and attending review . Get the CRITICAL_FINDINGS from the report." ,expected_output="reactivation tuberculosis necrotizing pneumonia", actual_output="['active tubercolosis']")
# correctness_metric = get_geval_metric(test_case)
# print(correctness_metric.score)
# print(correctness_metric.reason)


# 

##evaluating the models
# print ("Running G-Eval on all models ...")
# dict_allmodels_geval = dict()
# for model in all_models:
#     dict_allmodels_geval[model] = list()
#     geval_metric_score = list() #just to calculate the average and sd
#     model_answers =  data_df[model].tolist()
#     for i, item in enumerate(input_reports):
#         input_str = item + " The definition of Critical findings in a clinical report: CRITICAL findings are life threatening imaging findings that needs to be communicated immediately. Get the CRITICAL_FINDINGS from the report."
#         test_case = LLMTestCase(input=input_str, expected_output=ground_truth_answers[i], actual_output=model_answers[i])
#         correctness_metric = get_geval_metric(test_case)
#         tuple_vals = (correctness_metric.score, correctness_metric.reason)
#         dict_allmodels_geval[model].append(tuple_vals)
#         geval_metric_score.append(correctness_metric.score)
#     print ("Model: {}, Mean Score: {}, Std_Dev: {}".format(model, mean(geval_metric_score), stdev(geval_metric_score)))
# ##saving the scores and reasons in a file
# print ("Saving the G-Eval results to a file.")
# data_df_geval = pd.DataFrame.from_dict(dict_allmodels_geval)
# data_df_geval.to_csv("G_Eval_AllModels_MIMIC.csv", sep='\t')





