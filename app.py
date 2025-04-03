from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from typing import Dict, Any
import base64
import io
import os
import logging
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Railway Grievance API",
    description="API for categorizing railway grievances using AI and image analysis.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

categories_json = {
    "medical Assistance": ["medical assistance"],
    "Security": ["Eve-Teasing/Misbehaviour with lady passengers/Rape", "Theft of Passengers Belongings/Snatching", "Unauthorized person in Ladies/Disabled Coach/SLR/Reserve Coach",
                 "Harrasment/Extortion by security Personal/Railway personnel", "Nuisance by Hawkers/Beggar/Eunuch", "Luggage Left Behind/Unclaimed/Suspected Articles",
                 "Passenger Missing/Not Responding call", "Smoking/Drinking Alcohol/Narcotics", "Dacoity/Robbery/Murder/Riots", "Quarrelling/Hooliganism", 
                 "Passenger fallen down", "Nuisance by Passenger", "Misbehaviour", "Others"],
    "Handicapped Facilities": ["Handicapped Coach Facilities", "Handicapped toilet/washbasin", "Braille signage in coach", "Others"],
    "Facilities for Women with Special needs": ["Baby Food"],
    "Electrical Equipment": ["Air Conditioner", "Fans", "Lights", "Charging Points", "Others"],
    "coach-cleanliness": ["Toilet", "Washbasin", "Cockroach/Rodents", "Coach Interior", "Coach Exterior", "Others"],
    "Punctuality": ["NTES APP", "Late Running", "Others"],
    "Water Availability": ["Packaged Drinking Water/Rail Neer", "Toilet", "Washbasin", "Others"],
    "coach-maintenance": ["window/seat broken", "window/door locking problem", "tap leaking/tap not working problem", 
                     "broken/missing toilet fittings", "jerks/abnormal sounds", "other"],
    "Catering & Vending Services": ["Overcharging", "Service Quality & Hygiene", "Food Quality & Quantity", "E-Catering", "Food & Water Not Available", "Others"],
    "staff behavior": ["Staff Behaviour"],
    "Corruption/Bribery": ["Corruption/Bribery"],
    "Bed Roll": ["Dirty/Torn", "Overcharging", "Non Availability", "Others"]
}

class ImageInformation(BaseModel):
    category: str
    subcategory: str
    urgency: str
    preliminary_response: str

parser = JsonOutputParser(pydantic_object=ImageInformation)

def encode_and_compress_image(image: UploadFile, quality: int = 50, max_size: tuple = (100, 100)) -> str:
    try:
        img = Image.open(image.file)
        img.thumbnail(max_size)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing image")

@chain
def image_model(inputs: dict) -> Dict[str, Any]:
    model = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        api_key=openai_api_key,
    )

    msg = model.invoke([
        SystemMessage(
            content="You are a grievance response chatbot created to help users, that handles grievances related to Indian Railways given by the customers. Analyze the following image and description, and categorize the issue based on the categories provided."
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": inputs["prompt"]},
                {"type": "text", "text": parser.get_format_instructions()},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}"}
                },
            ]
        )
    ])

    response_text = msg.content.strip("```json").strip("```").strip() 
    try:
        return parser.parse(response_text)  
    except Exception as e:
        logger.error(f"Failed to parse AI response: {str(e)}")
        return {"error": "AI response parsing failed"}


@app.post("/analyze_issue/", response_model=ImageInformation)
async def analyze_issue(image: UploadFile = File(...), description: str = Form(...)):
    """
    Upload an image and provide a description of the issue.  
    The AI will analyze and categorize the complaint.
    """
    try:
        image_base64 = encode_and_compress_image(image)
        
        vision_prompt = f"""Description: "{description}",
        Categories and Subcategories:
        {categories_json}
        """

        vision_chain = image_model
        result = vision_chain.invoke(
            {
                "image": image_base64, 
                "prompt": vision_prompt
            }
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/")
def root():
    return {"message": "Railway Grievance API is running. Go to /docs to test."}
