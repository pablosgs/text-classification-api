# Text Classification API

This project provides a RESTful API for text classification using a machine learning model. The API is built with FastAPI and is designed to preprocess input text and perform inference to classify the text.

## Project Structure

```
text-classification-api
├── app
│   ├── main.py               # Entry point of the API
│   ├── models
│   │   └── model.py          # Model loading and management
│   ├── preprocessing
│   │   └── preprocess.py      # Text preprocessing functions
│   └── inference
│       └── infer.py          # Inference logic
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── .dockerignore              # Files to ignore in Docker build
├── .gitignore                 # Files to ignore in git
└── README.md                  # Project documentation
```

## Setup Instructions

### Using Docker

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pablosgs/text-classification-api
   cd text-classification-api
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t text-classification-api .
   ```

3. **Run the Docker container:**
   ```bash
   docker run -d -p 8080:8080 text-classification-api
   ```

4. **Test the endpoint:**
curl -X POST "http://localhost:8080/predict/" -H "Content-Type: application/json" -d '{"text": "Texto de ejemplo"}'

If you want to choose the model to be used for the classification, add to the request the parameter model.

  - For the embedding model: "Embedding" -> {"text": "Texto de ejemplo", "model": "Embedding"}
  - For the Gemini model: "Gemini" -> {"text": "Texto de ejemplo", "model": "Gemini"}
  - For the GPT model: "GPT" -> {"text": "Texto de ejemplo", "model": "GPT"}
  - For the Baseline model: "Baseline" -> {"text": "Texto de ejemplo", "model": "Baseline"}

  If no model is chosen, the API will use the baseline model.

### Using Python

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd text-classification-api
   ```

2. **Create and activate a virutalenv**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Run the API**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8080
   ```

4. **Test the endpoint:**
Same as in the Docker version.



### Environment Variables: 

Note that in order to use the GPT or Gemini models, credentials are needed. In order to provide the credentials, create a .env file with the following env variables:

For the GPT model:
AZURE_OPENAI_API_KEY="api-key"
AZURE_OPENAI_ENDPOINT="endpoint"
AZURE_OPENAI_API_VERSION="api-version"
AZURE_OPENAI_CHAT_VERSION="chat-version"

For the Gemini model:
GOOGLE_APPLICATION_CREDENTIALS="path/to/credential-json"

### Endpoints

- **POST /predict**
  - Description: Classifies the input text.
  - Request Body: 
    ```json
    {
      "text": "Your input text here",
      "model": "Baseline"
    }
    **Disclaimer:** If the `model` parameter is not provided, the API will default to using the Baseline model for classification.
    **Disclaimer:** If the `model` parameter is "Embedding", a pretrained model is needed at the path "~/text-classification-api/app/models/saved_models/saved_model.h5". In order to train it, you can use the train_model function of TextClassificationModel. You would need a dataset to do it.
    ```
  - Response: 
    ```json
    {
      "prediction": "Predicted class label"
    }
    ```

## Requirements

- Python 3.7+
- Docker
