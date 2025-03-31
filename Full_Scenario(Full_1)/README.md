"""
========================================================================================
Inference PROCESS with dynamic change of PECS + Green Credit Policy Parameters
========================================================================================

**Description:**
This version covers the Topic: Green Credit Policy Impact on Renewable Energy Adoption
First run the script to create and save the trained model. 

python RCP_PECS_GCP_v09.py  --scenario 26 \
    --subsidy 0.15 \
    --loan_interest_rate 0.04 \
    --tax_incentive_rate 0.1 \
    --social_prestige_weight 0.2 \
    --physis_health_status 0.9 \
    --physis_labor_availability 0.7 \
    --emotion_stress_level 0.2 \
    --emotion_satisfaction 0.8 \
    --cognition_policy_incentives 0.6 \
    --cognition_information_access 0.7 \
    --social_social_influence 0.5 \
    --social_community_participation 0.6


Regarding the parameters, they depend on the scenario:

- Scenario 1: High subsidies and low-interest loans for PV installations.
 --subsidy
 --loan_interest_rate

- Scenario 2: Tax incentives for renewable energy production.
--tax_incentive_rate

- Scenario 3: Policy with limited financial support but social prestige benefits for early adopters.
--social_prestige_weight


The script can be executed by defining only some Green Credit Policy parameters while keeping other parameters at their default values. 
The script is designed to accept partial input, where unspecified parameters will fall back to their default values as defined in the argument parser.


It can also be executed like: 

python RCP_PECS_GCP_v09.py --scenario 26 or python RCP_PECS_GCP_v09.py --past

This will take the default parameters.

The above will create a folder called trained models and insode it will save the ppo model in .zip format 

and the initial pecs and green credit policy partameters in json. 

Once the model is saved, the user can provide modified PECS and Green Credit Policy parameters in json and run the following script: 

python run_with_modified_parameters_v04.py --scenario 26 or python run_with_modified_parameters_v04.py --past


This script executes a Reinforcement Learning simulation using a trained PPO (Proximal Policy Optimization) model 
within a customized Mesa environment (`MesaEnv`). It modifies PECS (Policy, Emotion, Cognition, Social) and Green Credit Policy parameters 
based on a provided JSON configuration file or default settings, aggregates environmental data for a specified RCP 
(Representative Concentration Pathway) scenario, and generates agents decisions that are saved in data/output/final_agent_decisions.csv

**Features:**
- Aggregates environmental data based on selected RCP scenarios.
- Modifies PECS parameters from a JSON configuration file.
- Automatically cleans up old logs and output files before execution.
- Executes simulations for a specified number of episodes.
- Captures and saves final agent decisions.
- Comprehensive logging for monitoring and debugging purposes.


"""

