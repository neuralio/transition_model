"""
==================================================
Inference PROCESS with dynamic change of PECS
==================================================

**Description:**
First run the script to create and save the trained model. 

python RCP_PECS_v11.py --scenario 26 \
    --physis_health_status 0.9 \
    --physis_labor_availability 0.7 \
    --emotion_stress_level 0.2 \
    --emotion_satisfaction 0.8 \
    --cognition_policy_incentives 0.6 \
    --cognition_information_access 0.7 \
    --social_social_influence 0.5 \
    --social_community_participation 0.6

It can also run like: 

python RCP_PECS_v11.py --scenario 26 (replace with the RCP Scenario) or python RCP_PECS_v11.py --past (for the past version)

This will take the defaults inside the script 

The above will create a folder called trained models and insode it will save the ppo model in .zip format 

and the initial pecs partameters in json. 

Once the model is saved, the user can provide modified pecs parameters in json and run the following script: 

python run_with_modified_pecs_v05.py \
    --scenario 26 \
    --main_dir "/path/to/input/data/of/model/" \
    --original_pecs_path "configs/original_pecs.json" \
    --output_dir "results/output" \
    --modified_pecs_path "configs/modified_pecs.json" \
    --model_path "models/ppo_trained.zip" \
    --num_episodes 10 \
    --log_file "logs/simulation.log" \
    --modifications_file "./modifications.json"

### The user can run also run  like this but hte paths have to be defined inside the script manually: 
python run_with_modified_pecs_v05.py --scenario 26 or python run_with_modified_pecs_v05.py --past

This script executes a Reinforcement Learning simulation using a trained PPO (Proximal Policy Optimization) model 
within a customized Mesa environment (`MesaEnv`). It modifies PECS (Policy, Emotion, Cognition, Social) parameters 
based on a provided JSON configuration file or default settings, aggregates environmental data for a specified RCP 
(Representative Concentration Pathway) scenario, and generates visualizations of agent decisions.

**Features:**
- Aggregates environmental data based on selected RCP scenarios.
- Modifies PECS parameters from a JSON configuration file.
- Automatically cleans up old logs and output files before execution.
- Executes simulations for a specified number of episodes.
- Captures and saves final agent decisions.
- Comprehensive logging for monitoring and debugging purposes.


"""

