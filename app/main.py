from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.preprocessing.preprocess import Preprocessing
from app.inference.infer import infer
import uvicorn
import os
os.environ['CURL_CA_BUNDLE'] = ''

class PredictRequest(BaseModel):
    text: str
    model: str

app = FastAPI()

@app.post("/predict/")
async def predict(request: PredictRequest):
    try:
        text = request.text
        model = request.model
        preprocessor = Preprocessing()
        preprocessed_text = preprocessor.preprocess_text(text)
        prediction = infer(preprocessed_text, request_model=model)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
