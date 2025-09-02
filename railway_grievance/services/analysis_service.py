import logging
from fastapi import UploadFile
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from ..schemas.grievance_schema import GrievanceAnalysis
from ..utils.image_processor import process_image
from ..core.config import settings

logger = logging.getLogger(__name__)

# The grievance categories data
categories_json = {
    "Medical Assistance": ["medical assistance"],
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


parser = JsonOutputParser(pydantic_object=GrievanceAnalysis)

def get_vision_chain() -> any:
    """
    Constructs and returns a LangChain (LCEL) chain for vision-based grievance analysis.
    This chain takes a user's description and an image, processes them, and returns
    a structured JSON output.
    """

    # Create a clean, human-readable string from the categories dictionary
    categories_list_str = ""
    for category, subcategories in categories_json.items():
        categories_list_str += f"\n- Category: '{category}'\n"
        categories_list_str += "  Subcategories:\n"
        for sub in subcategories:
            categories_list_str += f"    - '{sub}'\n"


    system_prompt = f"""You are an expert AI assistant for Indian Railways, responsible for categorizing passenger grievances with high accuracy.
Your task is to analyze the user's description and the provided image.
Based on this analysis, you must perform the following actions:
1.  Categorize the issue into one of the main categories and one of the corresponding subcategories from the list provided below.
2.  Assess the urgency of the issue and classify it as 'Low', 'Medium', or 'High'. For example, safety and medical issues are 'High' urgency.
3.  Formulate a brief, empathetic, and professional preliminary response to the passenger, acknowledging their complaint.

You MUST select a category and subcategory EXACTLY as they appear in this list. Do not create new ones. If a suitable subcategory is 'Others', use that.

Categories and Subcategories List:
{categories_list_str}

You MUST respond ONLY with a valid JSON object that adheres to the required format. Do not include any other text or explanations.
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", [
            {"type": "text", "text": "User Description: {description}"},
            {"type": "text", "text": "Format Instructions:\n{format_instructions}"},
            {
                "type": "image_url",
                "image_url": "data:image/jpeg;base64,{image_base64}"
            }
        ])
    ])
    
    # The model is initialized here, using the API key from our centralized settings
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0, # Set to 0.0 for more deterministic and reliable JSON output
        openai_api_key=settings.OPENAI_API_KEY,
        max_tokens=500
    )

    # This is the modern LangChain Expression Language (LCEL) way to build the chain.
    chain = (
        RunnablePassthrough()
        | prompt_template
        | model
        | parser
    )
    return chain

async def analyze_grievance(description: str, image: UploadFile) -> GrievanceAnalysis:
    """
    Orchestrates the grievance analysis process. This is the main function
    called by the API router.

    1. Processes the image to get a base64 string.
    2. Invokes the AI vision chain with the description and image data.
    3. Returns the structured analysis as a Pydantic object.
    """
    try:
        logger.info("Starting grievance analysis...")
        
        # Asynchronously process the image using the utility function
        image_base64 = await process_image(image)
        logger.info("Image processed successfully.")

        # Get the LangChain vision chain
        vision_chain = get_vision_chain()
        
        # Asynchronously invoke the chain
        logger.info("Invoking AI vision model...")
        result = await vision_chain.ainvoke({
            "description": description,
            "image_base64": image_base64,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info("AI analysis completed successfully.")
        
        return result

    except Exception as e:
        logger.error(f"An error occurred in the analysis service: {e}")
        raise