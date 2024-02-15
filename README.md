# CriticalFindings_extract
Instruction Tuned NLP Framework for Extracting Critical and Incidental Findings from Reports

We design an pre-trained Instuction Tuned Model-base Framework for automated extraction of Critical and Incidental findings from the Reports. 
In our framework, we use the Mistral-7B instruction tuned model and generate our responses in both a Zero-Shot or Few-Shot setting. 

Data:

The data we used is in the format of .CSV file with two columns - 'IMPRESSION' and 'FINDINGS'.

To run the code:

Step 1: Install all the required packages using the .txt file: 
        -- pip install -r /path/to/requirements.txt
        Alternatively, you can also install the packages in a virtual environment with a YML file. 
          -- conda env create -f environment_critfind.yml
        

Step 2: To get the Critical and Incidental Findings, change the run_model.sh file accordingly
        i. For a Zero-Shot setting (Default setting)
            The type '--type' parameter is set to ZS 
        ii. For a Few-Shot setting
            The type '--type' parameter is set to FS and prompts the user to input Few-Shot examples as a string.

       Please input the following parameters: Input File Name and Output File Name (both in CSV format)

Step 3: Run the model ./run_model.sh 
        The output will be saved as a CSV file in same directory - for example "generated_responses_MISTRAL_ZS.csv" 

Step 4: Extracting the key terms from Critical Findings list. 
        This uses an exact match algorithm, looking for substrings in the generated responses. 
        We provide a starting list of critical terms in the file "CriticalFindings.txt"
        The extracted key terms have been saved in CSV file, under the column 'CriticalFindingsTerms'. 
        
        -- python keyterm_extract.py generated_responses_MISTRAL_ZS.csv extracted_keyterms_MISTRAL_ZS.csv 
        
        

