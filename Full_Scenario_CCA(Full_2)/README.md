"""
===================================================================================================================
Inference PROCESS with dynamic change of PECS + Green Credit Policy Parameters +  Climate Change Adaptation 
===================================================================================================================

**Description:**
This version covers the Topic: Green Credit Policy Impact on Renewable Energy Adoption and Climate Change Adaptation for Agricultural and Energy Systems.
First run the script to create and save the trained model. 

python RCP_PECS_GCP_CCA_funds.py  --scenario 26 or python RCP_PECS_GCP_CCA_funds.py  --past

This will take the default parameters.

The above will create a folder called trained models and inside it will save the ppo model in .zip format 

and the initial pecs and green credit policy partameters in json. 

Once the model is saved, the user can provide modified PECS, Green Credit Policy parameters (modifications.json) and Climate losses (climate_losses.json) in json and run the following script: 

python run_with_modified_parameters_v05.py --scenario 26 or python run_with_modified_parameters_v05.py --past


This script executes a Reinforcement Learning simulation using a trained PPO (Proximal Policy Optimization) model 
within a customized Mesa environment (`MesaEnv`). It modifies PECS (Policy, Emotion, Cognition, Social), Green Credit Policy parameters and losses due to climate change based on a provided JSON configuration file or default settings, aggregates environmental data for a specified RCP 
(Representative Concentration Pathway) scenario, and generates agents decisions that are saved in data/output/final_agent_decisions.csv

"""

