from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import sys
sys.path.append('..')
from app.preprocessing.preprocess import Preprocessing
import tensorflow as tf
import numpy as np
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class TextClassificationModel:

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", num_classes: int = 2, trained_model: bool = True):
        """
        Initialize the TextClassificationModel with a Hugging Face embedding model and a simple neural network.
        
        Args:
            embedding_model_name (str): Name of the Hugging Face model for embeddings.
            num_classes (int): Number of output classes for classification.
        """

        self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
        
        # Define a simple 1-layer neural network for classification
        self.classification_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(384,)),  # Embedding size
            tf.keras.layers.Dense(num_classes, activation="softmax")  # Output layer for multiclass classification
        ])

        # Compile the model
        self.classification_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        if trained_model:
            self.classification_model.load_weights("app/models/saved_models/saved_model.h5")

        print("Initializing TextClassificationModel...")

    def predict(self, preprocessed_text: str):
        """
        Predict the class of the given preprocessed text.
        
        Args:
            preprocessed_text (str): The input text after preprocessing.
        
        Returns:
            int: The predicted class index.
        """
        # Generate embeddings using Hugging Face model and put them in the correct format
        embeddings = self.embedding_model.encode([preprocessed_text])
        embeddings = np.array(embeddings, dtype=np.float32)
        embeddings = tf.data.Dataset.from_tensor_slices((embeddings)).batch(1)
        
        tf.keras.backend.set_learning_phase(0)  # Ensure the model is in inference mode

        # Use the TensorFlow model for prediction
        prediction = self.classification_model.predict(embeddings)
        prediction_class = np.argmax(prediction, axis=1)[0]
        if prediction_class == 0:
            return "NEGATIVE"
        else:
            return "POSITIVE"
    
    def train_model(self, data_path):

        # Prepare the dataset
        train_data, val_data = self.prepare_dataset(data_path)
        print("Data prepared")

        for batch in train_data.take(1):
            inputs, labels = batch
            print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")

        # Use EarlyStopping with val_loss
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
        history = self.classification_model.fit(
            train_data,
            validation_data=val_data,
            epochs=200,
            verbose=1,
            callbacks=[callback]
        )
        print("Model trained")

        # Save the model weights
        self.classification_model.save_weights("~/text-classification-api/app/models/saved_models/saved_model.h5")
        print("Trained model saved")

        # Evaluate the model on the validation set
        val_loss, val_accuracy = self.classification_model.evaluate(val_data, verbose=1)
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {val_accuracy}")
        return history
    
    def prepare_dataset(self, csv_path):
            # Load the dataset
            data = pd.read_csv(csv_path)

            print("Data loaded")

            preprocessor = Preprocessing(
            lemmatize=False,
            remove_numbers=False,
            )

            data['text'] = preprocessor.preprocess_text_list(data['text'].to_list())

            print("Texts preprocessed")

            # Encode the labels
            label_encoder = LabelEncoder()
            data['category'] = data['prediction'].apply(lambda x: 0 if x == "NEGATIVE" else 1)

            print("Labels encoded")

            # Split the dataset into training and validation sets
            train_texts, val_texts, train_labels, val_labels = train_test_split(
            data['text'], data['category'], test_size=0.2, random_state=42
            )

            # Tokenize and convert texts to embeddings
            train_embeddings = []
            val_embeddings = []
            train_embeddings = self.embedding_model.encode(train_texts.to_list())
            val_embeddings = self.embedding_model.encode(val_texts.to_list())

            train_embeddings = np.array(train_embeddings, dtype=np.float32)
            val_embeddings = np.array(val_embeddings, dtype=np.float32)
            train_labels = np.array(train_labels, dtype=np.int32)
            val_labels = np.array(val_labels, dtype=np.int32)
    
            # Create TensorFlow datasets
            train_data = tf.data.Dataset.from_tensor_slices((train_embeddings, train_labels)).batch(32)
            val_data = tf.data.Dataset.from_tensor_slices((val_embeddings, val_labels)).batch(32)

            return train_data, val_data
    


class TextClassificationPipeline:

    def __init__(self):
        
        self.pipe = pipeline("text-classification")

    def predict(self, text):
        prediction = self.pipe(text)
        return prediction[0]['label']
    
if __name__=='__main__':
    
    os.environ['CURL_CA_BUNDLE'] = ''
    model = TextClassificationModel(trained_model=True)
    preprocessor = Preprocessing()
    text = preprocessor.preprocess_text("This restaurant is good")
    prediction = model.predict(text)
    print(prediction)