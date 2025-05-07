import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr, ValidationError, validator
import os
import tensorflow as tf
import pandas as pd
import json
from typing import Union, Optional, Literal
from stable_baselines3 import PPO

# Import Rate Limiting Libraries
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from geoserver_integration_pv_past import upload_2_geoserver_pv_past
from geoserver_integration_pv_future import upload_2_geoserver_pv_future
from geoserver_integration_base_rl_future import upload_2_geoserver_rl_future
from geoserver_integration_base_rl_past import upload_2_geoserver_rl_past
from geoserver_integration_pecs_rl_future import upload_2_geoserver_pecs_rl_future
from geoserver_integration_pecs_rl_past import upload_2_geoserver_pecs_rl_past
from geoserver_integration_full1_rl_future import upload_2_geoserver_full1_rl_future
from geoserver_integration_full1_rl_past import upload_2_geoserver_full1_rl_past
from geoserver_integration_full2_rl_future import upload_2_geoserver_full2_rl_future


from pv_module import (
    load_config,
    load_training_data,
    train_model,
    load_slope_data,
    load_rsds_data,
    load_temperature_data,
    predict_pv_suitability_past,
    apply_conditions_to_predictions,
    calculate_pv_yield_past,
    MAIN_DIR,
    TRAINING_DIR,
    PREDICTION_DIR
)

# Import FUTURE PV Module
from pv_module_future import (
    load_config as load_config_future,
    predict_pv_suitability_future,
    apply_conditions_to_predictions as apply_conditions_to_predictions_future,
    calculate_pv_yield_future,
    load_netcdf,  
    process_climate_data,  
    load_training_data, 
    train_model, 
    MAIN_DIR as MAIN_DIR_FUTURE,
    TRAINING_DIR as TRAINING_DIR_FUTURE,
    PREDICTION_DIR as PREDICTION_DIR_FUTURE,
    CLIMATE_PROJ_DIR,  # Already added
    SLOPE_DIR  # ADD THIS LINE
)

# ─────────────────────────────────────────────────────────────
#  BASE RL FUTURE (no name collisions)
# ─────────────────────────────────────────────────────────────
from base_RL_future import (
    test_check_env as base_future_test_check_env,
    train_rl_model as base_future_train_rl_model,
    evaluate_model as base_future_evaluate_model,
    get_file_paths as base_future_get_file_paths,
    load_pv_data   as base_future_load_pv_data,
    load_crop_data as base_future_load_crop_data,
    merge_data as base_future_merge_data,
    aggregate_data as base_future_aggregate_data,
    MesaEnv as BaseFutureEnv,
    RESULTS_DIR as BASE_FUTURE_RESULTS_DIR
)

# ─────────────────────────────────────────────────────────────
#  BASE RL PAST
# ─────────────────────────────────────────────────────────────
from base_RL_past import (
    test_check_env as base_past_test_check_env,
    train_rl_model as base_past_train_rl_model,
    evaluate_model as base_past_evaluate_model,
    get_file_paths as base_past_get_file_paths,
    load_pv_data as base_past_load_pv_data,
    load_crop_data as base_past_load_crop_data,
    merge_data as base_past_merge_data,
    aggregate_data as base_past_aggregate_data,
    MesaEnv as BasePastEnv,
    RESULTS_DIR as BASE_PAST_RESULTS_DIR
)

# ─────────────────────────────────────────────────────────────
#  PECS RL FUTURE
# ─────────────────────────────────────────────────────────────
from pecs_RL_future import (
    load_input_json as pecs_future_load_input_json,
    get_pecs_params as pecs_future_get_pecs_params,
    load_and_process_data as pecs_future_load_and_process_data,
    train_rl_model as pecs_future_train_rl_model,
    run_simulation as pecs_future_run_simulation,
    MesaEnv as PecsFutureEnv,
    check_env as pecs_future_check_env,
    RESULTS_DIR as PECS_FUTURE_RESULTS_DIR
)

# ─────────────────────────────────────────────────────────────
#  PECS RL PAST
# ─────────────────────────────────────────────────────────────
from pecs_RL_past import (
    load_input_json as pecs_past_load_input_json,
    get_pecs_params as pecs_past_get_pecs_params,
    load_and_process_data as pecs_past_load_and_process_data,
    train_rl_model as pecs_past_train_rl_model,
    run_simulation as pecs_past_run_simulation,
    MesaEnv as PecsPastEnv,
    check_env as pecs_past_check_env,
    RESULTS_DIR as PECS_PAST_RESULTS_DIR
)

# ─────────────────────────────────────────────────────────────
#  FULL1 RL FUTURE
# ─────────────────────────────────────────────────────────────
from full1_RL_future import (
    load_input_json as full1_future_load_input_json,
    create_aggregated_data as full1_future_create_aggregated_data,
    MesaEnv as Full1FutureEnv,
    check_env as full1_future_check_env,
    train_rl_phase as full1_future_train_rl_phase,
    simulation_rl_phase as full1_future_simulation_rl_phase,
    RESULTS_DIR as FULL1_FUTURE_RESULTS_DIR,
    DEFAULT_PECS_PARAMS as FULL1_DEFAULT_PECS_PARAMS,
    DEFAULT_SCENARIO_PARAMS as FULL1_DEFAULT_SCENARIO_PARAMS
)


# ─────────────────────────────────────────────────────────────
#  FULL1 RL PAST
# ─────────────────────────────────────────────────────────────
from full1_RL_past import (
    load_input_json as full1_past_load_input_json,
    create_aggregated_data as full1_past_create_aggregated_data,
    MesaEnv as Full1PastEnv,
    check_env as full1_past_check_env,
    train_rl_phase as full1_past_train_rl_phase,
    simulation_rl_phase as full1_past_simulation_rl_phase,
    RESULTS_DIR as FULL1_PAST_RESULTS_DIR,
    DEFAULT_PECS_PARAMS as FULL1_PAST_DEFAULT_PECS_PARAMS,
    DEFAULT_SCENARIO_PARAMS as FULL1_PAST_DEFAULT_SCENARIO_PARAMS,
    cleanup_files as full1_past_cleanup_files
)


# ─────────────────────────────────────────────────────────────
#  FULL2 RL FUTURE
# ─────────────────────────────────────────────────────────────
from full2_RL_future import (
    load_config as full2_future_load_config,
    train_and_simulate as full2_future_train_and_simulate,
    INPUT_JSON_PATH as FULL2_INPUT_JSON_PATH,
    MAIN_DIR as FULL2_MAIN_DIR,
    RESULTS_DIR as FULL2_RESULTS_DIR,
    MesaEnv as Full2FutureEnv,
    create_aggregated_data   as full2_future_create_aggregated_data,
    DEFAULT_PECS_PARAMS      as FULL2_DEFAULT_PECS_PARAMS,
    DEFAULT_SCENARIO_PARAMS  as FULL2_DEFAULT_SCENARIO_PARAMS,
)

# ─────────────────────────────────────────────────────────────
#  CONFIGURE SECURE LOGGING (OWASP-Compliant)
# ─────────────────────────────────────────────────────────────
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_file = "app.log"

file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CONFIGURE RATE LIMITER
# ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)  # Uses client's IP address for rate limiting
app = FastAPI()

# Register exception handler for rate limit exceeded
app.state.limiter = limiter
app.add_exception_handler(HTTPException, _rate_limit_exceeded_handler)

# Define file paths
# Define prediction file paths for PAST
PREDICTION_FILE_PAST = os.path.join(PREDICTION_DIR, "PAST_PV_SUITABILITY_PREDICTIONS.csv")
UPDATED_PREDICTION_FILE_PAST = os.path.join(PREDICTION_DIR, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")

# Define prediction file paths for FUTURE
PREDICTION_FILE_FUTURE = os.path.join(PREDICTION_DIR_FUTURE, "FUTURE_PV_SUITABILITY_PREDICTIONS.csv")
UPDATED_PREDICTION_FILE_FUTURE = os.path.join(PREDICTION_DIR_FUTURE, "FUTURE_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")


INPUT_JSON_PATH_PAST = os.path.join(MAIN_DIR, "input_mongo_past.json")
INPUT_JSON_PATH_FUTURE = os.path.join(MAIN_DIR, "input_mongo_pv_future.json")

INPUT_JSON_PATH_CROP_PAST = os.path.join(MAIN_DIR, "input_mongo_crop_past.json")
INPUT_JSON_PATH_CROP_FUTURE = os.path.join(MAIN_DIR, "input_mongo_crop_future.json")

INPUT_JSON_PATH_BASE_RL = os.path.join(MAIN_DIR, "input_mongo_base_future.json")
INPUT_JSON_PATH_BASE_RL_PAST = os.path.join(MAIN_DIR, "input_mongo_base_past.json")


INPUT_JSON_PATH_PECS_RL_FUTURE = os.path.join(MAIN_DIR, "input_mongo_pecs_future.json")
INPUT_JSON_PATH_PECS_RL_PAST = os.path.join(MAIN_DIR, "input_mongo_pecs_past.json")

INPUT_JSON_PATH_FULL1_RL_FUTURE = os.path.join(MAIN_DIR, "input_mongo_full1_future.json")
INPUT_JSON_PATH_FULL1_RL_PAST = os.path.join(MAIN_DIR, "input_mongo_full1_past.json")


INPUT_JSON_PATH_FULL2_RL_FUTURE = os.path.join(MAIN_DIR, "input_mongo_full2_future.json")

# ─────────────────────────────────────────────────────────────
#  OWASP INPUT VALIDATION: Pydantic Models
# ─────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    config_path: constr(strip_whitespace=True, regex=r"^[a-zA-Z0-9_\-/]+\.json$")
    data_path: constr(strip_whitespace=True, regex=r"^[a-zA-Z0-9_\-/]+\.csv$")


class PredictRequest(BaseModel):
    model_path: constr(strip_whitespace=True, regex=r"^[a-zA-Z0-9_\-/]+\.h5$") = os.path.join(TRAINING_DIR, "trained_model.h5")



class GeoServerUpdateRequest(BaseModel):
    workspace_name: constr(strip_whitespace=True, min_length=2, max_length=50)
    updated_layer_name_predictions: constr(strip_whitespace=True, min_length=2, max_length=100)
    updated_layer_name_yield: constr(strip_whitespace=True, min_length=2, max_length=100)


class PvPastRequest(BaseModel):
    model_path: constr(strip_whitespace=True, regex=r"^[a-zA-Z0-9_\-/]+\.h5$") = os.path.join(TRAINING_DIR, "trained_model.h5")
    config_path: str = INPUT_JSON_PATH_PAST
    data_path: constr(strip_whitespace=True, regex=r"^[a-zA-Z0-9_\-/]+\.csv$") = os.path.join(TRAINING_DIR, "PV_suitability_data_TRAIN.csv")


class PvFutureRequest(BaseModel):
    model_path: constr(strip_whitespace=True, regex=r"^[a-zA-Z0-9_\-/]+\.h5$") = os.path.join(TRAINING_DIR, "trained_model_future.h5")
    config_path: str = INPUT_JSON_PATH_FUTURE
    data_path: constr(strip_whitespace=True, regex=r"^[a-zA-Z0-9_\-/]+\.csv$") = os.path.join(TRAINING_DIR, "PV_suitability_data_TRAIN.csv")


class ModelParameters(BaseModel):
    period: str
    scenario: Optional[Union[int, Literal["past"]]] = None

    @validator("scenario", always=True)
    def validate_scenario(cls, value, values):
        period = values.get("period", "").lower()
        if period == "past":
            # For a past period, scenario must be "past" (or None, which is then set to "past")
            if value is None:
                return "past"
            if value != "past":
                raise ValueError("For a past period, scenario must be 'past'.")
        else:
            allowed = {26, 45, 85}
            if value not in allowed:
                raise ValueError(f"For a future period, scenario must be one of {allowed}")
        return value

class BaseRLInput(BaseModel):
    task_id: str
    user_id: str
    model_parameters: ModelParameters


class PecsRLFutureInput(BaseRLInput):
    @validator("model_parameters")
    def ensure_future(cls, model_params: ModelParameters):
        if model_params.period.strip().lower() != "future":
            raise ValueError("For the pecs_future endpoint, period must be 'future'")
        return model_params

class PecsRLPastInput(BaseRLInput):
    @validator("model_parameters")
    def ensure_past(cls, model_params: ModelParameters):
        if model_params.period.strip().lower() != "past":
            raise ValueError("For the pecs_past endpoint, period must be 'past'")
        return model_params

# New model for the full1_future endpoint:
class Full1RLFutureInput(BaseRLInput):
    @validator("model_parameters")
    def ensure_future_scenario(cls, model_params: ModelParameters):
        if model_params.period.strip().lower() != "future":
            raise ValueError("For the full1_future endpoint, period must be 'future'.")
        allowed = {26, 45, 85}
        if model_params.scenario not in allowed:
            raise ValueError(f"For a future period, scenario must be one of {allowed}.")
        return model_params


class Full1RLPastInput(BaseRLInput):
    @validator("model_parameters")
    def ensure_past_scenario(cls, model_params: ModelParameters):
        if model_params.period.strip().lower() != "past":
            raise ValueError("For the full1_past endpoint, period must be 'past'.")
        # The ModelParameters validator already converts a missing value to "past"
        # if period is "past", so no further check is needed.
        return model_params


class Full2RLFutureInput(BaseRLInput):
    @validator("model_parameters")
    def ensure_future_scenario(cls, model_params: ModelParameters):
        if model_params.period.strip().lower() != "future":
            raise ValueError("For the full2_future endpoint, period must be 'future'.")
        allowed = {26, 45, 85}
        if model_params.scenario not in allowed:
            raise ValueError(f"For a future period, scenario must be one of {allowed}.")
        return model_params


# ─────────────────────────────────────────────────────────────
#  OWASP: GLOBAL EXCEPTION HANDLER FOR VALIDATION ERRORS
# ─────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error occurred: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation Error: {exc.errors()} on {request.url}")
    return JSONResponse(status_code=422, content={"detail": "Invalid input data. Please check your request format."})


# ─────────────────────────────────────────────────────────────
#  TRAIN ENDPOINT (Rate Limited to 2 requests per minute)
# ─────────────────────────────────────────────────────────────
@app.post("/train")
@limiter.limit("2/minute")
async def train_endpoint(request: Request, request_body: TrainRequest):
    try:
        config = load_config(request.config_path)
        X_train, X_test, y_train, y_test = load_training_data(request.data_path)
        model, history = train_model(X_train, y_train)
        model.save(request.model_path)
        logger.info("Training completed successfully.")
        return {"status": "Training completed", "model_save_path": request.model_path, "history": history.history}
    except Exception as e:
        logger.error(f"Training Failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Training process failed.")


# ─────────────────────────────────────────────────────────────
#  PREDICT ENDPOINT (Rate Limited to 5 requests per minute)
# ─────────────────────────────────────────────────────────────
@app.post("/predict")
@limiter.limit("5/minute")
async def predict_endpoint(request: Request, request_body: PredictRequest = PredictRequest()):
    try:
        model = tf.keras.models.load_model(request.model_path)
        df_rsds_max = load_rsds_data()
        df_rsds_min = load_rsds_data()
        df_tas_mean = load_temperature_data()
        df_slope = load_slope_data()

        predictions_df, task_prediction_dir = predict_pv_suitability_past(
            model, df_rsds_max, df_rsds_min, df_tas_mean, df_slope
        )

        updated_df = apply_conditions_to_predictions(
            os.path.join(task_prediction_dir, "PAST_PV_SUITABILITY_PREDICTIONS.csv"),
            task_prediction_dir
        )

        df_yield, yield_file = calculate_pv_yield_past()

        return {
            "status": "Prediction and yield calculation completed",
            "predictions": updated_df.to_dict(orient="records"),
            "yield_data": df_yield.to_dict(orient="records"),
            "folder": task_prediction_dir
        }

    except Exception as e:
        logger.error(f"Prediction Failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────------
#  GEOSERVER UPDATE ENDPOINT (Rate Limited to 3 requests per minute)
# ─────────────────────────────────────────────────────────────------
@app.post("/geoserver_update")
@limiter.limit("3/minute")
async def geoserver_update(request: Request):
    try:
        # Upload data to GeoServer and retrieve updated layer info
        url, workspace_name, updated_layer_name_predictions, updated_layer_name_yield = upload_2_geoserver()

        # Prepare structured Geoserver data with WMS Legend URLs
        geoserver_data = {
            "PV Predictions Layer": f"{workspace_name}:{updated_layer_name_predictions}",
            "PV Predictions Layer // URL": f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_predictions}",
            "PV Yield Layer": f"{workspace_name}:{updated_layer_name_yield}",
            "PV Yield Layer // URL": f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_yield}"
        }

        # Load existing JSON file securely
        if os.path.exists(INPUT_JSON_PATH):
            with open(INPUT_JSON_PATH, "r", encoding="utf-8") as file:
                input_data = json.load(file)
        else:
            input_data = {}

        # Ensure 'results' field exists and update it with new data
        if "results" not in input_data:
            input_data["results"] = {}

        input_data["results"].update(geoserver_data)

        # Save updated JSON file securely
        with open(INPUT_JSON_PATH, "w", encoding="utf-8") as file:
            json.dump(input_data, file, indent=4)

        return {
            "status": "Geoserver Update completed",
            **geoserver_data
        }

    except Exception as e:
        logger.error(f"Geoserver Update Failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Geoserver update failed.")


# ─────────────────────────────────────────────────────────────
#  PV PAST ENDPOINT (Rate Limited to 2 requests per minute)
# ─────────────────────────────────────────────────────────────
@app.post("/pv_past")
@limiter.limit("2/minute")
async def pv_past(request: Request, request_body: PvPastRequest):
    try:

        # config, task_id, user_id = load_config(INPUT_JSON_PATH_PAST)
        config = load_config(INPUT_JSON_PATH_PAST)
        task_id = config.get("task_id")
        user_id = config.get("user_id")

        # 1️⃣ Train the model if it does not exist
        if not os.path.exists(request_body.model_path):  # ✅ Fix: Use request_body instead of request
            logger.info("No trained model found. Starting training process...")
            config = load_config(request_body.config_path)
            X_train, X_test, y_train, y_test = load_training_data(request_body.data_path)
            model, history = train_model(X_train, y_train)
            model.save(request_body.model_path)
            train_status = "Model trained successfully."
        else:
            train_status = "Existing trained model found. Skipping training."

        # 2️⃣ Load trained model
        model = tf.keras.models.load_model(request_body.model_path)

        # 3️⃣ Load required datasets for prediction
        df_rsds_max = load_rsds_data()
        df_rsds_min = load_rsds_data()
        df_tas_mean = load_temperature_data()
        df_slope = load_slope_data()

        # 4️⃣ Generate predictions and get task-specific directory
        predictions_df, task_prediction_dir = predict_pv_suitability_past(
            model, df_rsds_max, df_rsds_min, df_tas_mean, df_slope
        )

        # 5️⃣ Update predictions based on conditions
        updated_df = apply_conditions_to_predictions(
            os.path.join(task_prediction_dir, "PAST_PV_SUITABILITY_PREDICTIONS.csv"),
            task_prediction_dir
        )



        #updated_df = apply_conditions_to_predictions(task_prediction_dir, INPUT_JSON_PATH_PAST)


        # 6️⃣ Calculate PV Yield after predictions are finalized
        df_yield, yield_file = calculate_pv_yield_past()

        # 7️⃣ Upload results to GeoServer
        url, workspace_name, updated_layer_name_predictions, updated_layer_name_yield = upload_2_geoserver_pv_past()

        # 8️⃣ Prepare structured Geoserver data with WMS Legend URLs
        # geoserver_data = {
        #     "type": "WMS",
        #     "PV_Predictions_Layer": f"{workspace_name}:{updated_layer_name_predictions}",
        #     "PV_Predictions_Layer // URL": f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_predictions}",
        #     "PV_Yield_Layer": f"{workspace_name}:{updated_layer_name_yield}",
        #     "PV_Yield_Layer // URL": f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_yield}"
        # }


        geoserver_data = {
            "type": "WMS",
            "layers": [f"{workspace_name}:{updated_layer_name_predictions}",f"{workspace_name}:{updated_layer_name_yield}"]
        }

        # 9️⃣ Load existing JSON file securely
        if os.path.exists(INPUT_JSON_PATH_PAST):
            with open(INPUT_JSON_PATH_PAST, "r", encoding="utf-8") as file:
                input_data = json.load(file)
        else:
            input_data = {}

        # 🔹 Ensure 'results' field exists and update it with new data
        if "results" not in input_data:
            input_data["results"] = {}

        input_data["results"].update(geoserver_data)

        # 🔹 Save updated JSON file securely
        with open(INPUT_JSON_PATH_PAST, "w", encoding="utf-8") as file:
            json.dump(input_data, file, indent=4)

        # ✅ 10️⃣ Return a consolidated response
        return {
            "status": "PV Past workflow completed successfully.",
            # "train_status": train_status,
            # "predictions": updated_df.to_dict(orient="records"),
            # "yield_data": df_yield.to_dict(orient="records"),
            "geoserver_data": geoserver_data,
            "task_folder": task_prediction_dir  # Return task folder for reference
        }

    except Exception as e:
        logger.error(f"PV Past Workflow Failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="PV Past workflow failed.")


# ─────────────────────────────────────────────────────────────
#  PV FUTURE ENDPOINT (Rate Limited to 2 requests per minute)
# ─────────────────────────────────────────────────────────────
@app.post("/pv_future")
@limiter.limit("2/minute")
async def pv_future(request: Request, request_body: PvFutureRequest):
    try:
        
        # config, task_id, user_id = load_config(INPUT_JSON_PATH_PAST)
        config = load_config(INPUT_JSON_PATH_FUTURE)
        task_id = config.get("task_id")
        user_id = config.get("user_id")

        # Ensure scenario exists in the JSON
        if "scenario" not in config["model_parameters"]:
            raise KeyError("The 'scenario' key is missing from 'model_parameters' in the JSON file.")

        scenario = config["model_parameters"]["scenario"]
        logger.info(f"Running PV Future workflow for Scenario RCP{scenario}...")

        # 2.Define Model File Path
        model_path = os.path.join(TRAINING_DIR, f"trained_model_rcp{scenario}.h5")

        # 3. Train Model if it doesn't exist
        if not os.path.exists(model_path):
            logger.info(f"No trained model found for RCP{scenario}. Training new model...")
            X_train, X_test, y_train, y_test = load_training_data(os.path.join(TRAINING_DIR, "PV_suitability_data_TRAIN.csv"))
            model, history = train_model(X_train, y_train, config)
            model.save(model_path)
            train_status = f"Model trained successfully for RCP{scenario}."
        else:
            train_status = f"Existing trained model found for RCP{scenario}. Skipping training."

        # 4. Load Trained Model
        model = tf.keras.models.load_model(model_path)

        # 5. Load Future Climate Datasets
        df_rsds_max = process_climate_data(CLIMATE_PROJ_DIR, "rsds", scenario)
        df_rsds_min = process_climate_data(CLIMATE_PROJ_DIR, "rsds", scenario)
        df_tas_max = process_climate_data(CLIMATE_PROJ_DIR, "tas", scenario)
        df_tas_min = process_climate_data(CLIMATE_PROJ_DIR, "tas", scenario)
        df_slope = process_climate_data(SLOPE_DIR, "slope", scenario)

        # 6. Generate Predictions
        predictions_df, task_prediction_dir = predict_pv_suitability_future(
            model, df_rsds_max, df_rsds_min, df_tas_max, df_tas_min, df_slope, INPUT_JSON_PATH_FUTURE
        )

        # 7. Apply Constraints and Adjust Predictions

        updated_df = apply_conditions_to_predictions_future(task_prediction_dir, INPUT_JSON_PATH_FUTURE)

        # 8. Calculate PV Yield
        df_yield, yield_file = calculate_pv_yield_future(INPUT_JSON_PATH_FUTURE)

        # 9. Upload Results to GeoServer
        url, workspace_name, updated_layer_name_predictions, updated_layer_name_yield = upload_2_geoserver_pv_future()

        # 10. Prepare Structured Geoserver Data with WMS Legend URLs
        # geoserver_data = {
        #     "type": "WMS",
        #     "PV_Predictions_Layer": f"{workspace_name}:{updated_layer_name_predictions}",
        #     "PV_Predictions_Layer // URL": f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_predictions}",
        #     "PV_Yield_Layer": f"{workspace_name}:{updated_layer_name_yield}",
        #     "PV_Yield_Layer // URL": f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_yield}"
        # }


        geoserver_data = {
            "type": "WMS",
            "layers": [f"{workspace_name}:{updated_layer_name_predictions}",f"{workspace_name}:{updated_layer_name_yield}"]
        }

        # 11. Load existing JSON file securely
        if os.path.exists(INPUT_JSON_PATH_FUTURE):
            with open(INPUT_JSON_PATH_FUTURE, "r", encoding="utf-8") as file:
                input_data = json.load(file)
        else:
            input_data = {}

        # Ensure 'results' field exists and update it with new data
        if "results" not in input_data:
            input_data["results"] = {}

        input_data["results"].update(geoserver_data)

        # Save updated JSON file securely
        with open(INPUT_JSON_PATH_FUTURE, "w", encoding="utf-8") as file:
            json.dump(input_data, file, indent=4)

        # Return Final Consolidated Response
        return {
            "status": f"PV Future workflow completed for RCP{scenario}.",
            # "train_status": train_status,
            # "predictions": updated_df.to_dict(orient="records"),
            # "yield_data": df_yield.to_dict(orient="records"),
            "geoserver_data": geoserver_data,
            "task_folder": task_prediction_dir  # Return task folder for reference
        }

    except Exception as e:
        logger.error(f"PV Future Workflow Failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="PV Future workflow failed.")


# ─────────────────────────────────────────────────────────────
#  CROP PAST ENDPOINT (Rate Limited to 2 requests per minute)
# ─────────────────────────────────────────────────────────────
@app.post("/crop_past")
@limiter.limit("2/minute")
async def crop_past(request: Request):
    try:
        # Define GeoServer base URL inside the endpoint
        geoserver_url = "http://159.69.72.104:8080/geoserver"

        # Ensure input JSON file exists
        if not os.path.exists(INPUT_JSON_PATH_CROP_PAST):
            raise HTTPException(status_code=400, detail="Input JSON file for past crop suitability not found.")

        # Load input JSON
        with open(INPUT_JSON_PATH_CROP_PAST, "r", encoding="utf-8") as file:
            input_data = json.load(file)

        # Extract `crop_type` from JSON
        crop_type = input_data["model_parameters"].get("crop_type", "").upper()
        if crop_type not in ["MAIZE", "WHEAT"]:
            raise HTTPException(status_code=400, detail="Invalid crop type in JSON. Only MAIZE and WHEAT are supported.")

        # Construct the correct WMS Layer name
        workspace_name = "Land_Suitability"
        updated_layer_name_predictions = f"Crop_Suitability_Predictions_Past_{crop_type}"

        # Construct the WMS URL
        # geoserver_data = {
        #     "type": "WMS",
        #     "Crop_Predictions_Layer": f"{workspace_name}:{updated_layer_name_predictions}",
        #     "Crop_Predictions_Layer // URL": f"{geoserver_url}/ows?service=WMS&version=1.3.0"
        #                                       f"&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20"
        #                                       f"&layer={workspace_name}%3A{updated_layer_name_predictions}"
        # }



        geoserver_data = {
            "type": "WMS",
            "layers": [f"{workspace_name}:{updated_layer_name_predictions}"]
        }

        # Ensure `results` field exists in JSON
        if "results" not in input_data:
            input_data["results"] = {}

        # Update results and save back to JSON
        input_data["results"].update(geoserver_data)
        with open(INPUT_JSON_PATH_CROP_PAST, "w", encoding="utf-8") as file:
            json.dump(input_data, file, indent=4)

        return {
            "status": "Crop past suitability WMS URL generated.",
            "geoserver_data": geoserver_data
        }

    except Exception as e:
        logger.error(f"Crop Past Workflow Failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Crop past workflow failed.")

# ─────────────────────────────────────────────────────────────
#  CROP FUTURE ENDPOINT (Rate Limited to 2 requests per minute)
# ─────────────────────────────────────────────────────────────
@app.post("/crop_future")
@limiter.limit("2/minute")
async def crop_future(request: Request):
    try:
        # Define GeoServer base URL inside the endpoint
        geoserver_url = "http://159.69.72.104:8080/geoserver"

        # Ensure input JSON file exists
        if not os.path.exists(INPUT_JSON_PATH_CROP_FUTURE):
            raise HTTPException(status_code=400, detail="Input JSON file for future crop suitability not found.")

        # Load input JSON
        with open(INPUT_JSON_PATH_CROP_FUTURE, "r", encoding="utf-8") as file:
            input_data = json.load(file)

        # Extract `crop_type` and `scenario` from JSON
        crop_type = input_data["model_parameters"].get("crop_type", "").upper()
        scenario = input_data["model_parameters"].get("scenario")

        # Validate `crop_type` and `scenario`
        if crop_type not in ["MAIZE", "WHEAT"]:
            raise HTTPException(status_code=400, detail="Invalid crop type in JSON. Only MAIZE and WHEAT are supported.")
        if not isinstance(scenario, int) or scenario <= 0:
            raise HTTPException(status_code=400, detail="Invalid scenario in JSON. Must be a positive integer.")

        # Construct the correct WMS Layer name
        workspace_name = "Land_Suitability"
        updated_layer_name_predictions = f"Crop_Suitability_Predictions_RCP{scenario}_{crop_type}"


        # Construct the WMS URL
        # geoserver_data = {
        #     "type": "WMS",
        #     "Crop_Predictions_Layer": f"{workspace_name}:{updated_layer_name_predictions}",
        #     "Crop_Predictions_Layer // URL": f"{geoserver_url}/ows?service=WMS&version=1.3.0"
        #                                       f"&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20"
        #                                       f"&layer={workspace_name}%3A{updated_layer_name_predictions}"
        # }


        geoserver_data = {
            "type": "WMS",
            "layers": [f"{workspace_name}:{updated_layer_name_predictions}"]
        }

        # Ensure `results` field exists in JSON
        if "results" not in input_data:
            input_data["results"] = {}

        # Update results and save back to JSON
        input_data["results"].update(geoserver_data)
        with open(INPUT_JSON_PATH_CROP_FUTURE, "w", encoding="utf-8") as file:
            json.dump(input_data, file, indent=4)

        return {
            "status": f"Crop future suitability WMS URL generated for scenario {scenario}.",
            "geoserver_data": geoserver_data
        }

    except Exception as e:
        logger.error(f"Crop Future Workflow Failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Crop future workflow failed.")

# ─────────────────────────────────────────────────────────────
# Base RL Past Endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/base_past")
@limiter.limit("2/minute")
async def base_past_endpoint(request: Request):
    """
    Endpoint for executing the BASE RL Past workflow, using aliased imports
    """
    try:
        # Read and validate input JSON
        with open(INPUT_JSON_PATH_BASE_RL_PAST, "r") as f:
            input_data_raw = json.load(f)
        input_data = BaseRLInput(**input_data_raw)

        if input_data.model_parameters.period.strip().lower() != "past":
            raise Exception("This endpoint is only for past data (period must be 'past').")

        task_id = input_data.task_id
        user_id = input_data.user_id
        if not task_id or not user_id:
            raise Exception("The input JSON must contain both 'task_id' and 'user_id'.")

        # Create results directory
        results_subdir = os.path.join(BASE_PAST_RESULTS_DIR, f"{task_id}_{user_id}")
        os.makedirs(results_subdir, exist_ok=True)
        logger.info(f"Results will be saved in: {results_subdir}")

        # Load and prepare data
        file_paths = base_past_get_file_paths(scenario=None, past=True)
        PV_, PV_Yield_ = base_past_load_pv_data(
            file_paths["pv_suitability_file"],
            file_paths["pv_yield_file"]
        )
        crop_suitability, crop_profit = base_past_load_crop_data(
            file_paths["cs_maize_file"],
            file_paths["cs_wheat_file"],
            file_paths["cy_maize_file"],
            file_paths["cy_wheat_file"]
        )
        env_data = base_past_merge_data(crop_suitability, PV_, PV_Yield_, crop_profit)
        aggregated_data = base_past_aggregate_data(env_data)

        # Instantiate environment and run checks
        env = BasePastEnv(env_data=aggregated_data, max_steps=10)
        base_past_test_check_env("past", past=True)

        # Train if needed
        model_file      = os.path.join(results_subdir, "past_ppo_mesa_model")
        model_file_zip  = model_file + ".zip"
        if os.path.exists(model_file_zip):
            logger.info("Trained model already exists; skipping training step.")
        else:
            base_past_train_rl_model(env, "past", results_subdir)

        # Evaluate
        base_past_evaluate_model("past", env, results_subdir)
        
        # Upload Results to GeoServer.
        url, workspace_name, updated_layer_name_predictions = upload_2_geoserver_rl_past()
        
        # Prepare Structured Geoserver Data with WMS Legend URLs.
        # geoserver_data = {
        #     "type": "WMS",
        #     "BASE_RL_Layer": f"{workspace_name}:{updated_layer_name_predictions}",
        #     "BASE_RL_Layer // URL": f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_predictions}"
        # }

        geoserver_data = {
            "type": "WMS",
            "layers": [f"{workspace_name}:{updated_layer_name_predictions}"]
        }
        

        # Load existing JSON file securely.
        if os.path.exists(INPUT_JSON_PATH_BASE_RL_PAST):
            with open(INPUT_JSON_PATH_BASE_RL_PAST, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
        else:
            existing_data = {}
        
        # Ensure 'results' field exists and update it with new data.
        if "results" not in existing_data:
            existing_data["results"] = {}
        existing_data["results"].update(geoserver_data)
        
        # Save updated JSON file securely.
        with open(INPUT_JSON_PATH_BASE_RL_PAST, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, indent=4)
        
        # Return Final Consolidated Response.
        return JSONResponse(
            status_code=200,
            content={
                "status": "BASE RL Past workflow completed.",
                "geoserver_data": geoserver_data,
                "task_folder": results_subdir
            }
        )
    except Exception as e:
        logger.error(f"Base Past endpoint failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Base Past endpoint failed.")


# ─────────────────────────────────────────────────────────────
# Base RL Future Endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/base_future")
@limiter.limit("2/minute")
async def base_future_endpoint(request: Request):
    """
    Endpoint for executing the Base Future RL workflow, using aliased imports
    """
    try:
        # Read and validate input JSON
        with open(INPUT_JSON_PATH_BASE_RL, "r") as f:
            input_data_raw = json.load(f)
        input_data = BaseRLInput(**input_data_raw)

        scenario = input_data.model_parameters.scenario
        if scenario is None:
            raise Exception("The input JSON must contain a 'scenario' in model_parameters.")

        task_id = input_data.task_id
        user_id = input_data.user_id
        if not task_id or not user_id:
            raise Exception("The input JSON must contain both 'task_id' and 'user_id'.")

        # Create results directory
        results_subdir = os.path.join(BASE_FUTURE_RESULTS_DIR, f"{task_id}_{user_id}")
        os.makedirs(results_subdir, exist_ok=True)
        logger.info(f"Results will be saved in: {results_subdir}")

        # Load and prepare data
        file_paths = base_future_get_file_paths(scenario, past=False)
        PV_, PV_Yield_ = base_future_load_pv_data(
            file_paths["pv_suitability_file"],
            file_paths["pv_yield_file"]
        )
        crop_suitability, crop_profit = base_future_load_crop_data(
            file_paths["cs_maize_file"],
            file_paths["cs_wheat_file"],
            file_paths["cy_maize_file"],
            file_paths["cy_wheat_file"]
        )
        env_data = base_future_merge_data(crop_suitability, PV_, PV_Yield_, crop_profit)
        aggregated_data = base_future_aggregate_data(env_data)

        # Instantiate environment and run checks
        env = BaseFutureEnv(env_data=aggregated_data, max_steps=10)
        base_future_test_check_env(scenario)

        # Train if needed
        model_file      = os.path.join(results_subdir, f"RCP{scenario}_ppo_mesa_model")
        model_file_zip  = model_file + ".zip"
        if os.path.exists(model_file_zip):
            logger.info("Trained model already exists; skipping training step.")
        else:
            base_future_train_rl_model(env, scenario, results_subdir)

        # Evaluate
        base_future_evaluate_model(scenario, env, results_subdir)

        # Upload Results to GeoServer
        url, workspace_name, updated_layer_name_predictions = upload_2_geoserver_rl_future()

        # Prepare Structured Geoserver Data with WMS Legend URLs
        # geoserver_data = {
        #     "type": "WMS",
        #     "BASE_RL_Layer": f"{workspace_name}:{updated_layer_name_predictions}",
        #     "BASE_RL_Layer // URL": f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_predictions}"
        # }


        geoserver_data = {
            "type": "WMS",
            "layers": [f"{workspace_name}:{updated_layer_name_predictions}"]
        }

        # Load existing JSON file securely
        if os.path.exists(INPUT_JSON_PATH_BASE_RL):
            with open(INPUT_JSON_PATH_BASE_RL, "r", encoding="utf-8") as file:
                input_data = json.load(file)
        else:
            input_data = {}

        # Ensure 'results' field exists and update it with new data
        if "results" not in input_data:
            input_data["results"] = {}

        input_data["results"].update(geoserver_data)

        # Save updated JSON file securely
        with open(INPUT_JSON_PATH_BASE_RL, "w", encoding="utf-8") as file:
            json.dump(input_data, file, indent=4)

        # Return Final Consolidated Response
        return {
            "status": f"BASE RL Future workflow completed for RCP{scenario}.",
            "geoserver_data": geoserver_data,
            "task_folder": results_subdir  # Return task folder for reference
        }
    except Exception as e:
        logger.error(f"Base Future endpoint failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Base Future endpoint failed.")


# ─────────────────────────────────────────────────────────────
# PECS RL Past Endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/pecs_past")
@limiter.limit("2/minute")
async def pecs_past_endpoint(request: Request):
    """
    Endpoint for executing the PECS RL Past workflow using aliased imports
    """
    try:
        # Load and validate the input JSON
        input_data = pecs_past_load_input_json(INPUT_JSON_PATH_PECS_RL_PAST)
        validated = PecsRLPastInput(**input_data)

        # Ensure period is 'past'
        if validated.model_parameters.period.lower() != "past":
            raise HTTPException(status_code=400, detail="Period must be 'past' for this endpoint.")

        # Prepare results directory
        task_id = validated.task_id
        user_id = validated.user_id
        results_subdir = os.path.join(PECS_PAST_RESULTS_DIR, f"{task_id}_{user_id}")
        os.makedirs(results_subdir, exist_ok=True)

        # Extract PECS params
        pecs_params = pecs_past_get_pecs_params(input_data)

        # Load and aggregate environment data
        aggregated_data = pecs_past_load_and_process_data()
        if aggregated_data is None or aggregated_data.empty:
            raise Exception("Past environmental data could not be loaded or is empty.")

        # Create and check the environment
        env = PecsPastEnv(env_data=aggregated_data, max_steps=10,
                          pecs_params=pecs_params, width=10, height=10)
        pecs_past_check_env(env)

        # Train or load model
        model_file     = os.path.join(results_subdir, "past_ppo_mesa_model")
        model_file_zip = model_file + ".zip"
        if os.path.exists(model_file_zip):
            logger.info("Existing model found; skipping training.")
            model = PPO.load(model_file_zip, env=env)
        else:
            model = pecs_past_train_rl_model(env, results_subdir, scenario_str="past")

        # Run simulation
        pecs_past_run_simulation(env, model, results_subdir, num_episodes=5)

        # Perform GeoServer integration specifically for PECS RL Past
        url, workspace_name, updated_layer_name_predictions = upload_2_geoserver_pecs_rl_past()

        # geoserver_data = {
        #     "type": "WMS",
        #     "PECS_RL_Layer": f"{workspace_name}:{updated_layer_name_predictions}",
        #     "PECS_RL_Layer // URL": f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_predictions}"
        # }


        geoserver_data = {
            "type": "WMS",
            "layers": [f"{workspace_name}:{updated_layer_name_predictions}"]
        }

        # Update input JSON with geoserver data
        input_data.setdefault("results", {})
        input_data["results"].update(geoserver_data)
        with open(INPUT_JSON_PATH_PECS_RL_PAST, "w", encoding="utf-8") as f:
            json.dump(input_data, f, indent=4)

        return {
            "status": "PECS Past workflow completed.",
            "geoserver_data": geoserver_data,
            "task_folder": results_subdir
        }

    except Exception as e:
        logging.error(f"PECS Past endpoint failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="PECS Past workflow failed.")



# ─────────────────────────────────────────────────────────────
# PECS RL Future Endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/pecs_future")
@limiter.limit("2/minute")
async def pecs_future_endpoint(request: Request):
    """
    Endpoint for executing the PECS RL Future workflow using aliased imports
    """
    try:
        # Load and validate the input JSON
        input_data = pecs_future_load_input_json(INPUT_JSON_PATH_PECS_RL_FUTURE)
        validated = PecsRLFutureInput(**input_data)

        # Extract scenario
        scenario = validated.model_parameters.scenario
        scenario_str = f"RCP{scenario}"

        # Prepare results directory
        task_id = validated.task_id
        user_id = validated.user_id
        results_subdir = os.path.join(PECS_FUTURE_RESULTS_DIR, f"{task_id}_{user_id}")
        os.makedirs(results_subdir, exist_ok=True)

        # Extract PECS params
        pecs_params = pecs_future_get_pecs_params(input_data)

        # Load and aggregate environment data
        aggregated_data = pecs_future_load_and_process_data(scenario)
        if aggregated_data is None or aggregated_data.empty:
            raise Exception("Environmental data could not be loaded or is empty.")

        # Create and check the environment
        env = PecsFutureEnv(env_data=aggregated_data, max_steps=10,
                            pecs_params=pecs_params, width=10, height=10)
        pecs_future_check_env(env)

        # Train or load model
        model_file     = os.path.join(results_subdir, f"{scenario_str}_ppo_mesa_model")
        model_file_zip = model_file + ".zip"
        if os.path.exists(model_file_zip):
            logger.info("Existing model found; skipping training.")
            model = PPO.load(model_file_zip, env=env)
        else:
            model = pecs_future_train_rl_model(env, results_subdir, scenario_str)

        # Run simulation
        pecs_future_run_simulation(env, model, results_subdir, num_episodes=5)

        
        # Perform GeoServer integration for PECS RL Future
        url, workspace_name, updated_layer_name_predictions = upload_2_geoserver_pecs_rl_future()
        
        # geoserver_data = {
        #     "type": "WMS",
        #     "PECS_RL_Layer": f"{workspace_name}:{updated_layer_name_predictions}",
        #     "PECS_RL_Layer // URL": f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_predictions}"
        # }


        geoserver_data = {
            "type": "WMS",
            "layers": [f"{workspace_name}:{updated_layer_name_predictions}"]
        }



        # Update input JSON with geoserver data
        input_data.setdefault("results", {})
        input_data["results"].update(geoserver_data)
        with open(INPUT_JSON_PATH_PECS_RL_FUTURE, "w", encoding="utf-8") as f:
            json.dump(input_data, f, indent=4)
        
        return {
            "status": f"PECS Future workflow completed for {scenario_str}.",
            "geoserver_data": geoserver_data,
            "task_folder": results_subdir
        }
    except Exception as e:
        logging.error(f"PECS Future endpoint failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="PECS Future workflow failed.")
        
# ─────────────────────────────────────────────────────────────
# FULL1 RL Past Endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/full1_past")
@limiter.limit("2/minute")
async def full1_past_endpoint(request: Request):
    """
    Endpoint for executing the FULL1 RL Past workflow using aliased imports
    """
    try:
        # 1️⃣ Load & validate input JSON
        raw_input_data = full1_past_load_input_json(INPUT_JSON_PATH_FULL1_RL_PAST)
        validated = Full1RLPastInput.parse_obj(raw_input_data)

        # 2️⃣ Prepare results directory
        task_id = validated.task_id
        user_id = validated.user_id
        results_subdir = os.path.join(FULL1_PAST_RESULTS_DIR, f"{task_id}_{user_id}")
        os.makedirs(results_subdir, exist_ok=True)

        # 3️⃣ Build PECS + scenario params
        mp = validated.model_parameters.dict()
        pecs_params = {
            "physis":    mp.get("physis",    FULL1_PAST_DEFAULT_PECS_PARAMS["physis"]),
            "emotion":   mp.get("emotion",   FULL1_PAST_DEFAULT_PECS_PARAMS["emotion"]),
            "cognition": mp.get("cognition", FULL1_PAST_DEFAULT_PECS_PARAMS["cognition"]),
            "social":    mp.get("social",    FULL1_PAST_DEFAULT_PECS_PARAMS["social"]),
        }
        scenario_params = mp.get("scenario_params", FULL1_PAST_DEFAULT_SCENARIO_PARAMS)
        scenario_str = "PAST"

        # 4️⃣ Aggregate data & instantiate environment
        aggregated_data = full1_past_create_aggregated_data(scenario_str, MAIN_DIR, results_subdir)
        env = Full1PastEnv(
            env_data=aggregated_data,
            max_steps=10,
            pecs_params=pecs_params,
            scenario_params=scenario_params,
            width=10,
            height=10
        )
        full1_past_check_env(env)

        # 5️⃣ Train or load model (with early stopping already wired into train_rl_phase)
        model_path = os.path.join(results_subdir, "past_ppo_mesa_model.zip")
        if not os.path.exists(model_path):
            ppo_model, model_path = full1_past_train_rl_phase(
                env,
                results_subdir,
                scenario_str,
                pecs_params,
                scenario_params,
                eval_freq=1000,
                patience=5,
                min_delta=1e-2
            )
        else:
            ppo_model = PPO.load(model_path, env=env)

        # 6️⃣ Run simulation
        agent_data_file = full1_past_simulation_rl_phase(
            ppo_model,
            env,
            results_subdir,
            num_episodes=5
        )

        # 7️⃣ GeoServer integration
        url, workspace_name, updated_layer_name_predictions = upload_2_geoserver_full1_rl_past()
        geoserver_data = {
            "type": "WMS",
            "layers": [f"{workspace_name}:{updated_layer_name_predictions}"]
        }

        # 8️⃣ Update input JSON with GeoServer results
        raw_input_data.setdefault("results", {})
        raw_input_data["results"].update(geoserver_data)
        with open(INPUT_JSON_PATH_FULL1_RL_PAST, "w", encoding="utf-8") as f:
            json.dump(raw_input_data, f, indent=4)

        # 9️⃣ Return consolidated response
        return JSONResponse(
            status_code=200,
            content={
                "status": "Full1 Past workflow completed.",
                "geoserver_data": geoserver_data,
                "task_folder": results_subdir,
                "model_path": model_path,
                "agent_data_file": agent_data_file
            }
        )

    except Exception as e:
        logging.error(f"Full1 Past endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Full1 Past workflow failed.")

# ─────────────────────────────────────────────────────────────
# FULL1 RL Future Endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/full1_future")
@limiter.limit("2/minute")
async def full1_future_endpoint(request: Request):
    """
    Endpoint for executing the FULL1 RL Future workflow using aliased imports.
    """
    try:
        # 1️⃣ Load raw JSON so we can update it later
        raw_input_data = full1_future_load_input_json(INPUT_JSON_PATH_FULL1_RL_FUTURE)
        if not raw_input_data:
            raise HTTPException(status_code=400, detail="Input JSON not found or invalid.")

        # 2️⃣ Validate
        validated = Full1RLFutureInput.parse_obj(raw_input_data)

        # 3️⃣ Prepare results directory
        task_id = validated.task_id
        user_id = validated.user_id
        results_subdir = os.path.join(FULL1_FUTURE_RESULTS_DIR, f"{task_id}_{user_id}")
        os.makedirs(results_subdir, exist_ok=True)

        # 4️⃣ Extract PECS & scenario parameters
        mp = validated.model_parameters.dict()
        pecs_params = {
            "physis":    mp.get("physis",    FULL1_DEFAULT_PECS_PARAMS["physis"]),
            "emotion":   mp.get("emotion",   FULL1_DEFAULT_PECS_PARAMS["emotion"]),
            "cognition": mp.get("cognition", FULL1_DEFAULT_PECS_PARAMS["cognition"]),
            "social":    mp.get("social",    FULL1_DEFAULT_PECS_PARAMS["social"]),
        }
        scenario_params = mp.get("scenario_params", FULL1_DEFAULT_SCENARIO_PARAMS)
        scenario = validated.model_parameters.scenario
        scenario_str = f"RCP{scenario}"

        # 5️⃣ Aggregate data
        aggregated = full1_future_create_aggregated_data(scenario, MAIN_DIR, results_subdir)
        if aggregated is None:
            raise HTTPException(status_code=500, detail="Data aggregation failed.")

        # 6️⃣ Instantiate & check environment
        env = Full1FutureEnv(
            env_data=aggregated,
            max_steps=10,
            pecs_params=pecs_params,
            scenario_params=scenario_params,
            width=10,
            height=10
        )
        full1_future_check_env(env)

        # 7️⃣ Train (with early stopping) or load existing model
        model_path = os.path.join(results_subdir, f"{scenario_str}_ppo_mesa_model.zip")
        if not os.path.exists(model_path):
            ppo_model, model_path = full1_future_train_rl_phase(
                env,
                results_subdir,
                scenario,
                pecs_params,
                scenario_params,
                eval_freq=1000,
                patience=5,
                min_delta=1e-2
            )
        else:
            ppo_model = PPO.load(model_path, env=env)

        # 8️⃣ Run simulation episodes
        agent_data_file = full1_future_simulation_rl_phase(
            ppo_model,
            env,
            results_subdir,
            num_episodes=5
        )

        # 9️⃣ GeoServer integration (using the same integration function as in the pecs_future endpoint)
        url, workspace_name, updated_layer_name_predictions = upload_2_geoserver_full1_rl_future()
        # geoserver_data = {
        #     "type": "WMS",
        #     "Full1_RL_Layer": f"{workspace_name}:{updated_layer_name_predictions}",
        #     "Full1_RL_Layer // URL": f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_predictions}"
        # }
        geoserver_data = {
            "type": "WMS",
            "layers": [f"{workspace_name}:{updated_layer_name_predictions}"]
        }

        # 🔟 Update JSON with GeoServer results
        raw_input_data.setdefault("results", {})
        raw_input_data["results"].update(geoserver_data)
        with open(INPUT_JSON_PATH_FULL1_RL_FUTURE, "w", encoding="utf-8") as f:
            json.dump(raw_input_data, f, indent=4)

        # ✅ Return consolidated response
        return JSONResponse(
            status_code=200,
            content={
                "status":          f"Full1 Future workflow completed for {scenario_str}.",
                "geoserver_data":  geoserver_data,
                "task_folder":     results_subdir,
                "model_path":      model_path,
                "agent_data_file": agent_data_file
            }
        )

    except Exception as e:
        logger.error(f"Full1 Future endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Full1 Future workflow failed.")



# ─────────────────────────────────────────────────────────────
# FULL2 RL Future Endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/full2_future")
@limiter.limit("2/minute")
async def full2_future_endpoint(request: Request):
    """
    FULL2 RL Future workflow with two-stage logic (train/simulate then manual simulation if needed)
    """
    try:
        # 1️⃣ Load & validate configuration
        config_data = full2_future_load_config(FULL2_INPUT_JSON_PATH)
        validated   = Full2RLFutureInput(**config_data)
        config      = validated.dict()

        # 2️⃣ Prepare results directories
        task_id          = config.get("task_id", "default_task")
        user_id          = config.get("user_id", "default_user")
        results_task_dir = os.path.join(FULL2_RESULTS_DIR, f"{task_id}_{user_id}")
        os.makedirs(results_task_dir, exist_ok=True)

        # 3️⃣ Scenario string and subdir
        scenario      = str(config["model_parameters"]["scenario"])
        scenario_str  = f"RCP{scenario}"
        scenario_output_dir = os.path.join(results_task_dir, scenario_str)
        os.makedirs(scenario_output_dir, exist_ok=True)

        # 4️⃣ Paths for model & output CSV
        trained_models_dir = os.path.join(scenario_output_dir, "trained_models")
        os.makedirs(trained_models_dir, exist_ok=True)
        model_file      = os.path.join(trained_models_dir, f"{scenario_str}_ppo_mesa_model")
        agent_data_file = os.path.join(scenario_output_dir, "agent_data_over_time.csv")

        # 5️⃣ Train & simulate if model missing
        if not os.path.exists(model_file):
            logger.info("No pre-trained model found; running train_and_simulate.")
            full2_future_train_and_simulate(config)
        else:
            logger.info("Pre-trained model found; skipping train_and_simulate.")

        # 6️⃣ If CSV still missing, run manual simulation
        if not os.path.exists(agent_data_file):
            logger.info("Simulation output missing; running manual simulation.")

            # a) re-aggregate data
            agg = full2_future_create_aggregated_data(
                scenario, FULL2_MAIN_DIR, scenario_output_dir
            )
            if agg is None:
                logger.error("Data aggregation failed.")
                return JSONResponse(status_code=500, content={"detail": "Data aggregation failed."})

            # b) extract PECS & scenario params
            mp            = config["model_parameters"]
            pecs_params   = {
                k: mp.get(k, FULL2_DEFAULT_PECS_PARAMS[k])
                for k in FULL2_DEFAULT_PECS_PARAMS
            }
            scenario_params = mp.get("scenario_params", FULL2_DEFAULT_SCENARIO_PARAMS)
            climate_losses  = mp.get("climate_losses", {})

            # c) instantiate env
            env = Full2FutureEnv(
                env_data=agg,
                max_steps=10,
                pecs_params=pecs_params,
                scenario_params=scenario_params,
                climate_losses=climate_losses,
                width=10,
                height=10
            )

            # d) load model & simulate episodes
            ppo_model = PPO.load(model_file, env=env)
            all_decisions = []
            for ep in range(5):
                obs, _ = env.reset()
                done = False
                truncated = False
                while not done and not truncated:
                    action, _ = ppo_model.predict(obs, deterministic=True)
                    obs, _, done, truncated, _ = env.step(action)
                for agent in env.model.schedule.agents:
                    all_decisions.append({
                        "episode":   ep + 1,
                        "agent_id":  agent.unique_id,
                        "lat":       agent.lat,
                        "lon":       agent.lon,
                        "decision":  getattr(agent, "decision", None)
                    })

            # e) save CSV
            if all_decisions:
                import pandas as pd
                pd.DataFrame(all_decisions).to_csv(agent_data_file, index=False)
                logger.info(f"Saved manual simulation to {agent_data_file}")
            else:
                logger.warning("No agent decisions collected during manual simulation.")

        
        # Perform GeoServer integration.
        url, workspace_name, updated_layer_name_predictions = upload_2_geoserver_full2_rl_future(agent_data_file)
        # geoserver_data = {
        #     "type": "WMS",
        #     "Full2_RL_Layer": f"{workspace_name}:{updated_layer_name_predictions}",
        #     "Full2_RL_Layer // URL": (
        #         f"{url}/ows?service=WMS&version=1.3.0&request=GetLegendGraphic"
        #         f"&format=image%2Fpng&width=20&height=20&layer={workspace_name}%3A{updated_layer_name_predictions}"
        #     )
        # }


        geoserver_data = {
            "type": "WMS",
            "layers": [f"{workspace_name}:{updated_layer_name_predictions}"]
        }

        
        # Update the configuration JSON with GeoServer results.
        config.setdefault("results", {})
        config["results"].update(geoserver_data)
        with open(INPUT_JSON_PATH_FULL2_RL_FUTURE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        
        # Return the consolidated response.
        return JSONResponse(
            status_code=200,
            content={
                "status": f"Full2 Future workflow completed for {scenario_str}.",
                "geoserver_data": geoserver_data,
                "task_folder": scenario_output_dir,
                "model_path": model_file,
                "agent_data_file": agent_data_file
            }
        )
        
    except Exception as e:
        logger.error(f"Full2 Future endpoint failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Full2 Future workflow failed.")
