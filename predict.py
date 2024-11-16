# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import base64
from io import BytesIO
from cog import BasePredictor, Input, Path, BaseModel, Secret
from typing import List, Tuple, Dict, Any
import asyncio
from openai import OpenAI
from PIL import Image
import logging
import time
import zipfile
import os
import json

logger = logging.getLogger(__name__)

# Define constants
MODERATION_MODEL = "omni-moderation-latest"


class CategoryResult(BaseModel):
    category: str
    score: float


class SafetyCheckResult(BaseModel):
    input_reference: str
    is_safe: bool
    categories: List[CategoryResult]
    time_taken: float
    input_type: str


class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(
        self,
        images_zip: Path = Input(
            description="Optional ZIP file containing images to run safety checks on",
            default=None
        ),
        inputs: Path = Input(
            description="Optional JSON file of inputs, each can have 'text', 'image_url', or both",
            default=None
        ),
        api_key: Secret = Input(
            description="OpenAI API key",
        ),
    ) -> List[SafetyCheckResult]:
        """Run safety checks on images or text inputs"""
        start_time = time.time()

        self.api_key = api_key.get_secret_value()
        self.client = OpenAI(api_key=self.api_key)

        moderation_inputs = []
        input_references = []
        input_types = []

        # Process images from ZIP file if provided
        if images_zip is not None:
            images = self._extract_images(images_zip)
            for filename, image in images:
                img_str = self._image_to_base64(image)
                moderation_inputs.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"},
                })
                input_references.append(filename)
                input_types.append('image')

        # Process inputs if provided
        if inputs is not None:
            try:
                # Read the JSON content from the file
                with open(inputs, 'r') as f:
                    inputs_list = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse 'inputs' JSON: {e}")
                return []
            except Exception as e:
                logger.error(f"Failed to read 'inputs' file: {e}")
                return []

            for idx, item in enumerate(inputs_list):
                if 'text' in item and 'image_url' in item:
                    # Create separate input objects for text and image_url
                    input_obj_text = {
                        "type": "text",
                        "text": item['text']
                    }
                    moderation_inputs.append(input_obj_text)
                    input_references.append(item['text'])
                    input_types.append('text')

                    input_obj_image = {
                        "type": "image_url",
                        "image_url": {"url": item['image_url']}
                    }
                    moderation_inputs.append(input_obj_image)
                    input_references.append(item['image_url'])
                    input_types.append('image')
                elif 'text' in item:
                    input_obj = {
                        "type": "text",
                        "text": item['text']
                    }
                    moderation_inputs.append(input_obj)
                    input_references.append(item['text'])
                    input_types.append('text')
                elif 'image_url' in item:
                    input_obj = {
                        "type": "image_url",
                        "image_url": {"url": item['image_url']}
                    }
                    moderation_inputs.append(input_obj)
                    input_references.append(item['image_url'])
                    input_types.append('image')
                else:
                    logger.warning(f"No valid input in item at index {idx}")
                    continue  # Skip if no valid input

        # Run moderation on all inputs asynchronously
        results = asyncio.run(self._predict_async(moderation_inputs, input_references, input_types))

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
                    except IOError as e:
                        logger.error(f"Failed to open {filename}: {e}")
        return images

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    async def _predict_async(
        self,
        moderation_inputs: List[Dict[str, Any]],
        input_references: List[str],
        input_types: List[str]
    ) -> List[SafetyCheckResult]:
        # Limit the number of concurrent tasks using a semaphore
        semaphore = asyncio.Semaphore(5)  # Adjust based on rate limits

        tasks = [
            self.process_input(semaphore, input_obj, reference, input_type)
            for input_obj, reference, input_type in zip(moderation_inputs, input_references, input_types)
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def process_input(
        self,
        semaphore: asyncio.Semaphore,
        input_obj: Dict[str, Any],
        reference: str,
        input_type: str
    ) -> SafetyCheckResult:
        async with semaphore:
            is_safe, time_taken, categories = await self.run_safety_checker(input_obj)

        return SafetyCheckResult(
            input_reference=reference,
            is_safe=is_safe,
            categories=categories,
            time_taken=time_taken,
            input_type=input_type
        )

    async def run_safety_checker(self, input_obj: Dict[str, Any]) -> Tuple[bool, float, List[CategoryResult]]:
        try:
            start_time = time.time()

            moderation_input = [input_obj]

            # Call the moderation endpoint asynchronously
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.moderations.create(input=moderation_input, model=MODERATION_MODEL)
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
                            category=category, score=score
                        )
                    )

            return is_safe, end_time - start_time, categories

        except Exception as e:
            logger.error(f"An error occurred during safety checking: {e}")
            return True, 0.0, []
