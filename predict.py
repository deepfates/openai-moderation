# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import base64
from io import BytesIO
from cog import BasePredictor, Input, Path, BaseModel, Secret
from typing import List, Dict, Any, Tuple
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


class UnsafeInput(BaseModel):
    input_reference: str
    input_type: str
    categories: List[CategoryResult]
    time_taken: float


class ErrorResult(BaseModel):
    input_reference: str
    error: str
    time_taken: float


class Output(BaseModel):
    total_inputs: int
    total_time_taken: float
    average_time_per_input: float
    unsafe_inputs: List[UnsafeInput]
    errors: List[ErrorResult]


class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(
        self,
        images_zip: Path = Input(
            description="Optional ZIP file containing images to run safety checks on",
            default=None,
        ),
        inputs: Path = Input(
            description="Optional JSON file of inputs, each can have 'text', 'image_url', or both",
            default=None,
        ),
        api_key: Secret = Input(
            description="OpenAI API key",
        ),
    ) -> Output:
        """Run safety checks on images or text inputs."""
        start_time = time.time()
        self.api_key = api_key.get_secret_value()
        self.client = OpenAI(api_key=self.api_key)

        moderation_inputs = []
        input_references = []
        input_types = []

        # Process images from ZIP file if provided
        if images_zip is not None:
            images = self._extract_images(images_zip)
            for item in images:
                filename = item['filename']
                image = item['image']
                img_str = self._image_to_base64(image)
                moderation_inputs.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_str}"},
                    }
                )
                input_references.append(filename)
                input_types.append("image")

        # Process inputs if provided
        if inputs is not None:
            try:
                # Read the JSON content from the file
                with open(inputs, "r") as f:
                    inputs_list = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse 'inputs' JSON: {e}")
                raise ValueError(f"Failed to parse 'inputs' JSON: {e}")
            except Exception as e:
                logger.error(f"Failed to read 'inputs' file: {e}")
                raise ValueError(f"Failed to read 'inputs' file: {e}")

            for idx, item in enumerate(inputs_list):
                if "text" in item and "image_url" in item:
                    # For text input
                    input_obj_text = {
                        "type": "text",
                        "text": item["text"],
                    }
                    moderation_inputs.append(input_obj_text)
                    input_references.append(item["text"])
                    input_types.append("text")

                    # For image input
                    input_obj_image = {
                        "type": "image_url",
                        "image_url": {"url": item["image_url"]},
                    }
                    moderation_inputs.append(input_obj_image)
                    input_references.append(item["image_url"])
                    input_types.append("image")

                elif "text" in item:
                    input_obj = {
                        "type": "text",
                        "text": item["text"],
                    }
                    moderation_inputs.append(input_obj)
                    input_references.append(item["text"])
                    input_types.append("text")

                elif "image_url" in item:
                    input_obj = {
                        "type": "image_url",
                        "image_url": {"url": item["image_url"]},
                    }
                    moderation_inputs.append(input_obj)
                    input_references.append(item["image_url"])
                    input_types.append("image")

                else:
                    logger.warning(f"No valid input in item at index {idx}")
                    continue  # Skip if no valid input

        total_inputs = len(moderation_inputs)

        if total_inputs == 0:
            raise ValueError("No valid inputs provided.")

        # Run moderation on all inputs asynchronously
        results = asyncio.run(
            self._predict_async(moderation_inputs, input_references, input_types)
        )

        # Calculate total and average time based on individual input times
        total_time_taken = sum(result["time_taken"] for result in results)
        average_time_per_input = total_time_taken / total_inputs if total_inputs else 0

        end_time = time.time()
        wall_clock_time = end_time - start_time

        # Filter out safe inputs
        unsafe_results = [result for result in results if not result["is_safe"] and result["error"] is None]

        # Collect errors
        error_results = [result for result in results if result["error"] is not None]

        # Prepare the output
        output = Output(
            total_inputs=total_inputs,
            total_time_taken=wall_clock_time,
            average_time_per_input=average_time_per_input,
            unsafe_inputs=[
                UnsafeInput(
                    input_reference=result["input_reference"],
                    input_type=result["input_type"],
                    categories=result["categories"],
                    time_taken=result["time_taken"],
                )
                for result in unsafe_results
            ],
            errors=[
                ErrorResult(
                    input_reference=result["input_reference"],
                    error=result["error"],
                    time_taken=result["time_taken"],
                )
                for result in error_results
            ],
        )

        return output

    def _extract_images(
        self, images_zip: Path
    ) -> List[Dict[str, Any]]:
        images = []
        with zipfile.ZipFile(images_zip, "r") as zip_ref:
            for filename in zip_ref.namelist():
                # Skip directories and macOS metadata files
                if (
                    filename.endswith("/")
                    or filename.startswith("__MACOSX/")
                    or os.path.basename(filename).startswith("._")
                ):
                    continue
                with zip_ref.open(filename) as file:
                    image_data = file.read()
                    try:
                        image = Image.open(BytesIO(image_data)).convert("RGB")
                        images.append({"filename": filename, "image": image})
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
        input_types: List[str],
    ) -> List[Dict[str, Any]]:
        # Limit the number of concurrent tasks using a semaphore
        semaphore = asyncio.Semaphore(5)  # Adjust based on rate limits

        tasks = [
            self.process_input(semaphore, input_obj, reference, input_type)
            for input_obj, reference, input_type in zip(
                moderation_inputs, input_references, input_types
            )
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def process_input(
        self,
        semaphore: asyncio.Semaphore,
        input_obj: Dict[str, Any],
        reference: str,
        input_type: str,
    ) -> Dict[str, Any]:
        async with semaphore:
            start_time = time.time()
            try:
                is_safe, categories = await self.run_safety_checker(input_obj)
                error = None
            except Exception as e:
                logger.error(
                    f"An error occurred during safety checking for input {reference}: {e}"
                )
                is_safe = True  # Default to safe if an error occurs
                categories = []
                error = str(e)
            end_time = time.time()
            time_taken = end_time - start_time

        return {
            "input_reference": reference,
            "is_safe": is_safe,
            "categories": categories,
            "time_taken": time_taken,
            "input_type": input_type,
            "error": error,
        }

    async def run_safety_checker(
        self, input_obj: Dict[str, Any]
    ) -> Tuple[bool, List[CategoryResult]]:
        moderation_input = [input_obj]

        # Call the moderation endpoint synchronously in an executor
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.moderations.create(
                input=moderation_input, model=MODERATION_MODEL
            ),
        )

        result = response.results[0]
        is_safe = not result.flagged

        # Extract categories and scores
        categories = []
        category_flags = vars(result.categories)
        category_scores = vars(result.category_scores)

        for category, is_flagged in category_flags.items():
            if is_flagged:
                score = category_scores.get(category, 0.0)
                categories.append(
                    CategoryResult(category=category, score=score)
                )

        return is_safe, categories
