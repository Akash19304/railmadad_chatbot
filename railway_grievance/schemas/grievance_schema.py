from pydantic import BaseModel, Field

class GrievanceAnalysis(BaseModel):
    category: str = Field(description="The main category of the grievance from the provided list.")
    subcategory: str = Field(description="The subcategory of the grievance from the provided list.")
    urgency: str = Field(description="The assessed urgency level, must be one of: 'Low', 'Medium', 'High'.")
    preliminary_response: str = Field(description="A brief, empathetic initial response to the user acknowledging their complaint.")