##code to finetune mistral model on the data 
##ref: https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model-0f39647b20fe
## /home/avisha/Datacenter_storage/Critical_finding/Code_Avisha
## CUDA_VISIBLE_DEVICES=1,3 python finetuning_mistral.py
import os
import sys
import re
import json
from tqdm import tqdm
import pandas as pd
import torch
from datasets import Dataset, load_dataset
Dataset.cleanup_cache_files
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3" 

##reformatting the dataset 
## sample instance
# ##{
# "text":"<s>[INST] Create a function to calculate the sum of a sequence of integers. here are the inputs [1, 2, 3, 4, 5] [/INST]
# # Python code def sum_sequence(sequence): sum = 0 for num in sequence: sum += num return sum</s>",
# "instruction":"Create a function to calculate the sum of a sequence of integers",
# "input":"[1, 2, 3, 4, 5]",
# "output":"# Python code def sum_sequence(sequence): sum = 0 for num in sequence: sum += num return sum"
# }

def remove_newlinechars(my_string):
    return re.sub(r'\r\n', '', my_string)

def create_text_row(instruction, input, output):
    # this function is used to output the right formate for each row in the dataset
    text_row = f"""<s>[INST] {instruction} Here are the findings and response from the resport: {input} [/INST]. And the extracted terms for the critical findings are: {output} </s>"""
    return text_row

def process_input_file(data_df, output_file_path):
    ## get the data impression and findings 
    impression_txtlist = data_df['impression']
    findings_txtlist = data_df['finding']
    responses_txtlist = data_df['response']
    matched_termslist = data_df['Exact_Match_Terms']


    dict_data = [{'impression': impression, 'finding': finding, 'matched_terms': matched_terms, 'responses': responses} for impression, finding, matched_terms, responses in zip(impression_txtlist, findings_txtlist, matched_termslist, responses_txtlist)]
    print ("Number of instances loaded ",len(dict_data))

    ##the basic prompt with Critical Findings and Incidental Findings definition 
    hyp_base = " The definition of Critical and Incidental findings in a clinical report: CRITICAL findings are life threating imaging findings that needs to be communicated immediately while INCIDENTAL findings non-life-threatening findings, but significant enough that they need to be communicated within a short period of time."

    # ##concatenating the above definitions with further information of categories in CRITICAL findings
    # hyp_catCF = hyp_base + "CRITICAL findings can be further categorized into three types - NEW, KNOWN/EXPECTED, and UNCERTAIN. The following are the definitions of each of these. NEW: A new critical finding is a critical finding identified in the report to be present for the first time regardless if it is mild/minimal/small in severity, or a critical finding that is implied to have been previously reported but appears to be worsening or increasing in severity. The new critical finding category does not encompass critical findings that are stable, improving, or decreasing in severity. Such critical findings should be classified as known/expected. KNOWN/EXPECTED: A known/expected critical finding is a critical finding that is implied by the report to be known and is described as being either unchanged, improving, or decreasing in severity. This category does not include critical findings that are known but are worsening or increasing in severity; such findings should be categorized as new critical findings. This category also covers critical findings which may be anticipated as part of post-surgical changes. UNCERTAIN: An uncertain critical finding is a critical finding that isn't definitively confirmed in the report. These critical findings are mentioned as possibly being present, mentioned that they should be considered, or concern/suspicion regarding their presence are raised."

    instruction = f"{hyp_base} Based on these above definitions, find the CRITICAL FINDINGS TERMS mentioned in the report."

    with open(output_file_path, "w") as output_jsonl_file:
        for x in tqdm(range(len(dict_data))):
            context = remove_newlinechars(dict_data[x]['impression'] + dict_data[x]['finding'])
            json_object = {
                "text": create_text_row(instruction, context, dict_data[x]['responses'].rstrip()),
                "instruction": instruction,
                "input": context,
                "output": remove_newlinechars(dict_data[x]['matched_terms'])
            }
            output_jsonl_file.write(json.dumps(json_object))
            output_jsonl_file.write("\n")

def main():
    ## reading the dataset 
    file_path = "../phase_1/criticalterms_mimic_testdata_MISTRAL_ZS.csv.csv"
    
    print ("Loading the data set...{}".format(file_path))
    data_df = pd.read_csv(file_path)
    print ("Data Shape: ", data_df.shape) 
    print ("Data Columns: ", data_df.columns)

    ## creating a JSON file for processing
    process_input_file(data_df, "./weaktraining_mimic_testdata_MISTRAL_ZS.json")
    print ("processed the data to json.")


    ## loading the training
    print ("loading the training data.")

    # # basePath = os.path.dirname(os.path.abspath(__file__))
    file_path = "./weaktraining_mimic_testdata_MISTRAL_ZS.json"
    df = pd.read_json(open(file_path, "r", encoding="utf8"), lines=True)
    df.head()
    findings_dataset = Dataset.from_pandas(df)
    train_dataset = findings_dataset.shuffle(seed=666).select(range(3))
    print (train_dataset.shape)
    # print (train_dataset[0])

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    ## setting model parameters
    new_model = "mistralai-Code-Instruct-mimic" #set the name of the new model
    ################################################################################
    # QLoRA parameters
    ################################################################################
    lora_r = 16 # LoRA attention dimension
    lora_alpha = 4 # Alpha parameter for LoRA scaling
    lora_dropout = 0.1 # Dropout probability for LoRA layers

    ################################################################################
    # bitsandbytes parameters
    ################################################################################
    use_4bit = True # Activate 4-bit precision base model loading
    bnb_4bit_compute_dtype = "float16" # Compute dtype for 4-bit base models
    bnb_4bit_quant_type = "nf4" # Quantization type (fp4 or nf4)
    use_nested_quant = False # Activate nested quantization for 4-bit base models (double quantization)

    ################################################################################
    # TrainingArguments parameters
    ################################################################################
    output_dir = "./results_15K_FS" # Output directory where the model predictions and checkpoints will be stored
    num_train_epochs = 1 # Number of training epochs
    fp16 = False # Enable fp16/bf16 training (set bf16 to True with an A100)
    bf16 = False
    per_device_train_batch_size = 1 # Batch size per GPU for training
    per_device_eval_batch_size = 1 # Batch size per GPU for evaluation
    gradient_accumulation_steps = 1 # Number of update steps to accumulate the gradients for
    gradient_checkpointing = True # Enable gradient checkpointing
    max_grad_norm = 0.3 # Maximum gradient normal (gradient clipping)
    learning_rate = 2e-4 # Initial learning rate (AdamW optimizer)
    weight_decay = 0.001 # Weight decay to apply to all layers except bias/LayerNorm weights
    optim = "paged_adamw_32bit" # Optimizer to use
    lr_scheduler_type = "constant" # Learning rate schedule (constant a bit better than cosine)
    max_steps = -1 # Number of training steps (overrides num_train_epochs)
    warmup_ratio = 0.03 # Ratio of steps for a linear warmup (from 0 to learning rate)
    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True
    save_steps = 100 # Save checkpoint every X updates steps
    logging_steps = 25 # Log every X updates steps

    ################################################################################
    # SFT parameters
    ################################################################################

    max_seq_length = None # Maximum sequence length to use
    packing = False # Pack multiple short examples in the same input sequence to increase efficiency
    # device_id = "0" 
    # device = "cuda:{}".format(device_id)

    # device_map = {"": 3} # Load the entire model on the GPU 0


    ##Loading the base model
    print ("loading the model for finetuning.")

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)  # Load the base model with QLoRA configuration
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map = "auto",
        )
    
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Load MitsralAi tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ##starting the finetuning with qLora and Supervised Finetuning
    print ("starting model Fine-Tuning with qLora and Supervised Finetuning.")

    # Set LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=1000, # the total number of training steps to perform
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )

    # Initialize the SFTTrainer for fine-tuning
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,  # You can specify the maximum sequence length here
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )



    print ("start the model training.")
    print (training_arguments.device)
    # with torch.autocast("cuda"): 
    train_result = trainer.train() # Start the training process    
    ##writing the result of the training 
    with open("training_result_MISTRAL_FS_15K.txt", "w") as f:
        for line in trainer.state.log_history:
            f.write(f"{line}\n")

    trainer.model.save_pretrained(new_model) # Save the fine-tuned model
    print (train_result)

    ## testing the model inference
    # eval_prompt = """Print hello world in python c and c++"""
    eval_prompt = """Here are the findings and response from the resport: \nInterval advancement of disease with: new and enlarging pulmonarynodules, a new left epicardial nodule, new 1.3 hepatic metastasis in hepaticsegment IVb, indeterminate splenic lesion which could represent a small infarctor metastasis, enlarging infiltrating and vascular encasing/occluding pancreaticmass, new left para-aortic infiltrating soft tissue and findings most consistentwith peritoneal carcinomatosis including paracolic nodules and a potentialSister. Mary Joseph node in the umbilicus.MR abdomen: Bilateral lung base nodules measuring up to approximately 1.2 cm.New 1 cm epicardial nodule (series 28/image 22). Please see the separate chestCT report. Tiny foci of abdominal fluid with new subcentimeter enhancing fociabout the liver and spleen (series 28/images 22, 41, 42). New 1.3 x 1 cmdiffusion restricting lesion in hepatic segment IVb (series 27/image 56) likelyrepresenting a metastatic focus. Cholecystectomy. Mild bilateral intrahepaticbiliary ductal prominence. No significant extrahepatic ductal dilatation. Noobvious choledocholithiasis. New 1.3 x 1.3 cm subcapsular wedge-shaped area ofdecreased vascularity in the spleen (series 28/image 35). Subcentimeter softtissue nodules are seen adjacent to the spleen (series 28/image 22).Interval enlargement of the ill-defined pancreatic body mass currently measuring4 x 2.9 cm (series 28/image 46). Atrophic pancreatic tail. Again seen isencasement of the superior mesenteric artery and distal celiac axis. Again seenis irregularity of the proximal hepatic and splenic arteries. Stable occlusionof the superior mesenteric vein at the junction with the portal vein withcollateral vessels. Stable splenic vein occlusion.The kidneys are unremarkable. Atherosclerotic nonaneurysmal abdominal aorta.Interval development of ill-defined left para-aortic/left paralumbar soft tissuemeasuring 1.6 x 1.5 cm (series 28/image 56). No evidence for bowel obstruction.Multiple enhancing soft tissue nodules adjacent to the ascending and descendingcolon measuring up to 1.1 cm (series 28/images 63-75). Potential Sister. MaryJoseph 1 cm umbilical lymph node (series 28/image 70). Degenerative changes ofthe spine. And the extracted terms for the critical findings are: "[ """
    model_input = tokenizer(eval_prompt, return_tensors="pt").input_ids.to("cuda:0")
    # input_ids = tokenizer.encode(eval_prompt+"\n\nANSWER:", return_tensors='pt', return_attention_mask=False).to("cuda:0")
    trainer.model.eval()
    # # print (model_input.dtype)
    # for param in trainer.model.parameters():
    #     if param.dtype == torch.bfloat16:
    #         param.data.to(torch.float16)
    # sys.exit(1)
    with torch.no_grad():
        generated_code = tokenizer.decode(trainer.model.generate(model_input, max_new_tokens=100, pad_token_id=2)[0], skip_special_tokens=True)
        # generated_code = trainer.model.generate(input_ids, max_new_tokens=512, do_sample=True)
        # response = tokenizer.decode(generated_code[0], skip_special_tokens=True)
    print(generated_code)

if __name__ == "__main__":
    main()