import pandas as pd
import sys
import os

drifter_velocity_models_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
+ '/drifter_velocity_models')
sys.path.append(drifter_velocity_models_dir)
from drifter_velocity_models.linear_regression_model import LinearRegressionModel as m
from rmse_class import RMSE
from sems_class import SEMs
from aoems_class import AOEMs

data = pd.read_hdf("./drifter_velocity_models/ocean_data.h5")
covs = ['u_av','v_av','lat','lon','Tx','Ty','Wx','Wy','t']

model = m(data,covs)

def model_info(model_type):
# for linear regression models
    if model_type == "LR":
        model_details = f"Covariates: {covs}"
    else:
        model_details = ""
    return model_details

def results_row(model_type,model):
    # for errors
    preds = model.predictions
    obs = model.observations
    
    err1 = RMSE(preds,obs)
    err2 = SEMs(preds,obs)
    err3 = AOEMs(preds,obs)

    result_row = {"Model":model_type,"Model Details":model_info(model_type),
                  "RMSE":err1.error, "RMSE std":err1.uncertainty,
                  "MAE":err2.mae_speed.error, "MAE std":err2.mae_speed.uncertainty,
                  "MAE Over":err2.ma_overestimated_e.error, "MAE Over std":err2.ma_overestimated_e.uncertainty,
                  "MAE Under":err2.ma_underestimated_e.error, "MAE Under std":err2.ma_underestimated_e.uncertainty,
                  "Over/Under/Correct Speed": err2.over_under_correct_proportions,
                  "MAAO":err3.maao_all.error, "MAAO std":err3.maao_all.uncertainty,
                  "MAAO AC": err3.maao_anticlockwise.error, "MAAO AC std": err3.maao_anticlockwise.uncertainty,
                  "MAAO C": err3.maao_clockwise.error, "MAAO C std": err3.maao_clockwise.uncertainty,
                  "AC/C/None/Undefined":err3.clockwise_anticlockwise_no_undefined_proportions
                  }
    return result_row

print(results_row("LR",model))

    


