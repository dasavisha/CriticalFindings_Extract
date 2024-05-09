# CriticalFindings_extract
Instruction Tuned NLP Framework for Extracting Critical and Incidental Findings from Reports

We design an pre-trained Instuction Tuned Model-base Framework for automated extraction of Critical and Incidental findings from the Reports. 
In our framework, we use the Mistral-7B instruction tuned model and generate our responses in both a Zero-Shot or Few-Shot setting. 

Data:

The data we used is in the format of .CSV file with two columns - 'IMPRESSION' and 'FINDINGS'.

To run the code:

Step 1: Install all the required packages in a virtual environment. 
        You can install the packages using the requirements.txt file inside an anaconda environment. 
        
        conda create -n crit_find anaconda python=3.10
        pip install -r requirements.txt
        

Step 2: To extract the Critical Findings, run the bash file accordingly. 

i. For a Zero-Shot setting (Default setting), the type '--prompt_type' parameter is set to ZS 

ii. For a Few-Shot setting, the type '--prompt_type' parameter is set to FS and prompts the user to input Few-Shot examples as a string.

        >> Enter the Few-Shot Examples:
    
    Following is a User Input Example.
        >> "{
        "explanation": "Small bowel obstruction due to a closed loop obstruction is a critical finding because it is life-threatening if not treated immediately. It is characterized as new,                          because it is the noted to be present for the first time. Intestinal ischemia is a critical finding, because it is a medical emergency, needing to be acted on       
                        immediately. It is characterized as an uncertain critical finding because the possibility is raised.",
        "report": "Findings are consistent with small-bowel obstruction likely secondary to a closed loop obstruction. Additionally, presence of abdominal ascites, mesenteric engorgement,                       and differential enhancement of the wall raise concern for intestinal ischemia. These findings were discussed with Dr. [**Redacted Name**] on [**2107-11-21**].",
        "new": ["small bowel obstruction","closed loop obstruction"],
        "known/expected": [],
        "uncertain": ["intestinal ischemia"],
        }"

    To run the extraction phase with MISTRAL model: 
    
      ./run_model_phase1.sh

    
        
        

