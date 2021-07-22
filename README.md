# OUsToN
 
The script Ouston.py is the implementation of the Onine User-Topic Network model.
For more information, see the main article: _Online User-Topic Network recovery: what is spread, how it spreads, who spreads it_

## Usage
Run the script Ouston.py using the following syntax: [parameter]=[value][space]

The parameter are: data_file, output_folder, r, theta0, particle_num, printRes
- Mandatory parameters: data_file, output_folder
    - data_file: relative path to the file containing events. The file must be formatted so that each line follows this syntax:
        > index(int)[TAB]timestamp(float)[TAB]textual content(comma separated strings)[TAB]node_index(int)[TAB]cascade_number(int)[TAB]true_cluster(optional, int)[end_line]
    - output_folder: where to save output files
- Optional parameters: r, theta0, particle_num, printRes
    - r (default 1): what version of the Powered Dirichlet Process to use for its survival version. r=1 is the regular Dirichlet process, and r=0 reduces Ouston to TopicCascade.
    - theta0 (default 0.01): hyperparameter for the language model's Dirichlet prior
    - particle_num (default 4): how many particles the SMC algorithm will use
    - printRes (default True): whether to print progress every 100 treated observations. If the true cluster is provided, also displays NMI. If the true network is present in the input file directory, also attempts to compute the MAE on networks edges.
  
## Requirements
To run the model:
- python~=3.7
- numpy~=1.21.0
- sparse~=0.11.2
- cvxpy~=1.1.12
- scipy~=1.7.0
  
For data generation and visualization (Generate_data.py and Analyze.py):
- matplotlib~=3.0.2
- networkx~=2.4
- multidict~=5.1.0
- wordcloud~=1.8.1
- scikit-learn~=0.23.1