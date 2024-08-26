from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import base64
from PIL import Image
from io import BytesIO
import requests
import json
import re
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_api_key = os.environ.get("openai_api_key")

categories_json = {
    "medical Assisance": ["medical assistance"],
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

def encode_image(image_data, max_size=(200, 200), quality=75):
    image = Image.open(BytesIO(image_data))
    
    if image.format != "JPEG":
        image = image.convert("RGB")
    
    image.thumbnail(max_size)
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.post("/analyze-grievance/")
async def analyze_grievance(description: str, file: UploadFile = File(...)):

    image_data = await file.read()
    
    base64_image = encode_image(image_data)

    prompt = f"""
    Analyze the following image and description, and categorize the issue based on the categories provided.

    Image (base64): "{base64_image}"

    Description: "{description}"

    Categories and Subcategories:
    {categories_json}

    Return only a JSON response in this format:
    {{
        "category": "category_name",
        "subcategory": "subcategory_name",
        "severity": "high/low",
        "preliminary_response": "Provide a brief, empathetic response that acknowledges the complaint and outlines immediate action. For example, for a dirty toilet: 'Thank you for notifying us. We will send a cleaner right away.'"
    }}
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error processing the request")

    content_string = response.json()["choices"][0]["message"]["content"]

    json_match = re.search(r'\{.*\}', content_string, re.DOTALL)
    if json_match:
        content_json = json.loads(json_match.group())
        return content_json

    raise HTTPException(status_code=500, detail="No valid JSON object found in the response")
