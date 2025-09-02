from fastapi import UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image
import base64
import io
import logging

logger = logging.getLogger(__name__)

async def process_image(image: UploadFile, quality: int = 50, max_size: tuple = (128, 128)) -> str:
    """
    Asynchronously resizes, compresses, and base64 encodes an uploaded image.
    """
    def _process():
        try:
            img = Image.open(image.file)
            img.thumbnail(max_size)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise IOError("Failed to process image file. It might be corrupted or in an unsupported format.") from e

    return await run_in_threadpool(_process)