# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib 
# import numpy as np
# import uvicorn
# import os
# from dotenv import load_dotenv

# load_dotenv()

# try:
#     model = joblib.load("model.pkl")
#     scaler = joblib.load("scaler.pkl")
# except Exception as e:
#     raise RuntimeError(f"Error loading model or scaler: {e}")


# app = FastAPI(
#     title="Wine Quality Prediction API",
#     description="Predicts wine quality category (Best, Good, Average, Bad) based on chemical features.",
#     version="1.0.0"
# )


# class WineInput(BaseModel):
#     fixed_acidity: float
#     volatile_acidity: float
#     citric_acid: float
#     residual_sugar: float
#     chlorides: float
#     free_sulfur_dioxide: float
#     total_sulfur_dioxide: float
#     density: float
#     pH: float
#     sulphates: float
#     alcohol: float


# quality_map = {
#     9: "Best",
#     8: "Best",
#     7: "Good",
#     6: "Good",
#     5: "Average",
#     4: "Bad",
#     3: "Bad"
# }



# @app.get("/")
# def welcome():
#     """
#     Welcome route to confirm API is running.
#     """
#     return {"message": "Welcome to the Wine Quality Prediction API! Go to /docs for testing."}


# @app.post("/predict")
# def predict_wine_quality(data: WineInput):
#     """
#     Predicts wine quality based on input chemical properties.
#     """
#     try:
        
#         input_data = np.array([[ 
#             data.fixed_acidity,
#             data.volatile_acidity,
#             data.citric_acid,
#             data.residual_sugar,
#             data.chlorides,
#             data.free_sulfur_dioxide,
#             data.total_sulfur_dioxide,
#             data.density,
#             data.pH,
#             data.sulphates,
#             data.alcohol
#         ]])

        
#         scaled_data = scaler.transform(input_data)

        
#         pred = model.predict(scaled_data)[0]

        
#         quality_label = quality_map.get(pred, "Unknown")

#         return {
#             "prediction": int(pred),
#             "quality_label": quality_label
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
        
# if __name__ == "__main__":
#       print(os.getenv("host"))
#       print(os.getenv("port"))
#       uvicorn.run(app, host=os.getenv("host"), port=int(os.getenv("port")))

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from dotenv import load_dotenv
import uvicorn
import os
load_dotenv()
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")
app = FastAPI(
    title="Wine Quality Prediction API",
    description="Predicts wine quality category (Best, Good, Average, Bad) based on chemical features.",
    version="1.0.0"
)
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
quality_map = {
    9: "Best",
    8: "Best",
    7: "Good",
    6: "Good",
    5: "Average",
    4: "Bad",
    3: "Bad"
}
@app.get("/")
def welcome():
    """
    Welcome route to confirm API is running.
    """
    return {"message": "Welcome to the Wine Quality Prediction API! Go to /docs for testing."}
@app.post("/predict")
def predict_wine_quality(data: WineInput):
    """
    Predicts wine quality based on input chemical properties.
    """
    try:
        input_data = np.array([[
            data.fixed_acidity,
            data.volatile_acidity,
            data.citric_acid,
            data.residual_sugar,
            data.chlorides,
            data.free_sulfur_dioxide,
            data.total_sulfur_dioxide,
            data.density,
            data.pH,
            data.sulphates,
            data.alcohol
        ]])
        scaled_data = scaler.transform(input_data)
        pred = model.predict(scaled_data)[0]
        quality_label = quality_map.get(pred, "Unknown")
        return {
            "prediction": int(pred),
            "quality_label": quality_label
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
if __name__ == "__main__":
    print(os.getenv("host"))
    print(os.getenv("port"))
    uvicorn.run(app, host=os.getenv("host"), port=int(os.getenv("port")))