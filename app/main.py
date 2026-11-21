# FastAPI app for serving the model
import joblib
from fastapi import FastAPI, HTTPException

# Load model
model = joblib.load("/app/model.joblib")

app = FastAPI()


@app.post("/predict")
def predict(pixels: list[int]):

    # Check input dimensions
    if len(pixels) != 64:
        raise HTTPException(
            status_code=400,
            detail="Expected exactly 64 pixel values"
        )

    # Check pixel values
    if any(pixel < 0 or pixel > 16 for pixel in pixels):
        raise HTTPException(
            status_code=400,
            detail="Pixel values must be between 0 and 16"
        )

    # Predict
    prediction = model.predict([pixels])[0]

    return {"prediction": int(prediction)}
