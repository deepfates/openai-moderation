# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import base64
from io import BytesIO
from cog import BasePredictor, Input, Path, BaseModel, Secret
from typing import List, Tuple
import asyncio
from openai import OpenAI
from PIL import Image
import logging
import time
import functools
import zipfile
import io
import os

logger = logging.getLogger(__name__)


class CategoryResult(BaseModel):
    category: str
    is_flagged: bool
    score: float


class SafetyCheckResult(BaseModel):
    image_filename: str
    is_safe: bool
    categories: List[CategoryResult]
    time_taken: float


class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(
        self,
        images_zip: Path = Input(
            description="ZIP file containing images to run safety checks on"
        ),
        api_key: Secret = Input(
            description="OpenAI API key",
        ),
    ) -> List[SafetyCheckResult]:
        """Run safety checks on images in a ZIP file"""
        start_time = time.time()

        self.api_key = api_key.get_secret_value()
        self.client = OpenAI(api_key=self.api_key)

        # Extract images from the ZIP file
        images = self._extract_images(images_zip)

        # Run the asynchronous processing
        results = asyncio.run(self._predict_async(images))

        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")

        return results

    def _extract_images(self, images_zip: Path) -> List[Tuple[str, Image.Image]]:
        images = []
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            for filename in zip_ref.namelist():
                # Skip directories and macOS metadata files
                if (
                    filename.endswith('/')
                    or filename.startswith('__MACOSX/')
                    or os.path.basename(filename).startswith('._')
                ):
                    continue
                with zip_ref.open(filename) as file:
                    image_data = file.read()
                    try:
                        image = Image.open(BytesIO(image_data)).convert("RGB")
                        images.append((filename, image))
                    except Exception as e:
                        logger.error(f"Failed to open {filename}: {e}")
        return images

    async def _predict_async(self, images: List[Tuple[str, Image.Image]]) -> List[SafetyCheckResult]:
        tasks = [
            self.process_image(filename, image) for filename, image in images
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def process_image(self, filename: str, image: Image.Image) -> SafetyCheckResult:
        is_safe, time_taken, categories = await self.run_safety_checker(image)

        return SafetyCheckResult(
            image_filename=filename,
            is_safe=is_safe,
            categories=categories,
            time_taken=time_taken,
        )

    async def run_safety_checker(self, image: Image.Image):
        try:
            start_time = time.time()
            buffered = BytesIO()
            image.save(buffered, format="PNG")

            img_str = base64.b64encode(buffered.getvalue()).decode()

            moderation_input = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"},
                }
            ]

            # Use functools.partial to pass keyword arguments
            create_moderation = functools.partial(
                self.client.moderations.create,
                input=moderation_input,
                model="omni-moderation-latest",
            )

            # Call the moderation endpoint asynchronously
            response = await asyncio.get_event_loop().run_in_executor(
                None, create_moderation
            )

            end_time = time.time()

            result = response.results[0]
            is_safe = not result.flagged

            # Access categories and scores
            categories_obj = result.categories
            scores_obj = result.category_scores

            # Extract category names from the Categories object
            category_names = vars(categories_obj).keys()

            categories = []
            for category in category_names:
                is_flagged = getattr(categories_obj, category)
                score = getattr(scores_obj, category)
                if is_flagged:
                    categories.append(
                        CategoryResult(
                            category=category, is_flagged=is_flagged, score=score
                        )
                    )

            return is_safe, end_time - start_time, categories

        except Exception as e:
            logger.error(f"An error occurred during safety checking: {e}")
            return True, 0.0, []
