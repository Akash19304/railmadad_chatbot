import logging
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import Annotated

from ..core.security import get_api_key
from ..schemas.grievance_schema import GrievanceAnalysis
from ..services import analysis_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/analyze_issue/",
    response_model=GrievanceAnalysis,
    dependencies=[Depends(get_api_key)]
)
async def handle_analyze_issue(
    image: Annotated[UploadFile, File(description="Image of the issue, max 10MB.", media_type="image/jpeg")],
    description: Annotated[str, Form(min_length=10, max_length=1000)]
):
    """
    Upload an image and a description to analyze and categorize a railway grievance.
    """
    try:
        return await analysis_service.analyze_grievance(description=description, image=image)
    except IOError as e:
        logger.warning(f"Bad image upload attempt: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred in the analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")