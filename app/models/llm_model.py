from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_vertexai.chat_models import ChatVertexAI
from google.oauth2 import service_account
from dotenv import load_dotenv
import os
import re
import json
load_dotenv()

class ClassExtraction(BaseModel):
    #doctype can only be agency, company or prospectus
    clase: str = Field(description="""Retrieve the sentiment of the review. Retrieve ONLY ONE of the following:
                        - POSITIVE
                        - NEGATIVE
                        
                        Retrieve just the word for the sentiment.""", default= False)
    

class TextClassificationLLM:

    def __init__(self, model):

        self.model = model


    def text_classification_message(self, text):
        # Construct a refined message with clearer distinctions
        message = HumanMessage(
            content=f"""
                Interpret the review given in the following text:

                Text: {text}
            """
        )
        return message
    
    
    def load_model(self):

        if self.model == "GPT":
            
            llm = AzureChatOpenAI(
                    max_tokens=5000,
                    temperature=0,
                    max_retries=10,
                    request_timeout=500,
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_VERSION"),
                    )
        elif self.model == "Gemini":

            credentials = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
            flashModel = 'gemini-2.0-flash'
            llm = ChatVertexAI(
                    max_output_tokens = 5000, 
                    temperature = 0, 
                    model_name= flashModel,
                    credentials=credentials
            )

        return llm


    def predict(self, text):
        
        llm = self.load_model()
        message = self.text_classification_message(text)
        parser = JsonOutputParser(pydantic_object=ClassExtraction) 
        prompt = ChatPromptTemplate.from_messages([
            ("system", 'You are an expert in interpreting reviews.'
                'Return a valid JSON object ONLY. Do not include explanations or extra text. Format: {format_instructions}'),
            (message),
        ])
        chain = prompt | llm | parser
        raw_response = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        })

        if isinstance(raw_response, str):
            cleaned_response = re.sub(r',\s*(\}|\])', r'\1', raw_response)  
        elif isinstance(raw_response, dict):
            cleaned_response = json.dumps(raw_response)  
        else:
            cleaned_response = raw_response  

        try:
            response = parser.parse(cleaned_response)
        except Exception as e:
            print("Salida raw:", raw_response)
            print("Salida limpia:", cleaned_response)
            raise e

        return response['clase']