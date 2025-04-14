from fastapi import APIRouter
import sys
sys.path.append('..')
from app.models.model import TextClassificationModel, TextClassificationPipeline
from app.models.llm_model import TextClassificationLLM
from app.preprocessing.preprocess import Preprocessing
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['CURL_CA_BUNDLE'] = ''

router = APIRouter()

@router.post("/infer")
def infer(text: str, request_model: str = "Baseline"):

    if request_model == "Embedding":
        print("Using embedding model")
        model = TextClassificationModel(trained_model=True)
        prediction = model.predict(text)
    
    elif request_model == "Gemini" or request_model == "GPT":
        print(f"Using {request_model} model")
        model = TextClassificationLLM(model=request_model)
        prediction = model.predict(text)

    else:
        print("Using baseline model")
        pipe = TextClassificationPipeline()
        prediction = pipe.predict(text)

    return prediction

def annotate_data(data, pipe, preprocessor):

    data['prediction'] = ""
    for index, row in data.iterrows():
        print(f'{index}/{len(data)}')
        text = row['text']
        text = preprocessor.preprocess_text(text)
        data.at[index, 'prediction'] = pipe.predict(text)

    output_path = "~/text-classification-api/app/assets/sentiment_texts_annotated.csv"
    data.to_csv(output_path, index=False)


if __name__== '__main__':

    data = pd.read_csv("~/text-classification-api/app/assets/sentiment_texts.csv")
    pipe = TextClassificationPipeline()
    preprocessor = Preprocessing()
    annotate_data(data, pipe, preprocessor)