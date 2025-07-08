import importlib.metadata
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.models.base_model import BasePageModel, BaseVlmModel
from docling.models.layout_model import LayoutModel
from docling.models.utils.hf_model_download import (
    HuggingFaceModelDownloadMixin,
)
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class TwoStageVlmModel(BasePageModel, HuggingFaceModelDownloadMixin):
    def __init__(
        self,
        *,
        layout_model: LayoutModel,
        vlm_model: BaseVlmModel,
    ):
        self.layout_model = layout_model
        self.vlm_model = vlm_model

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "two-staged-vlm"):
                    assert page.size is not None

                    page_image = page.get_image(
                        scale=self.vlm_model.scale, max_size=self.vlm_model.max_size
                    )

                    pred_clusters = self.layout_model.predict_on_page(page_image=page_image)
                    page, processed_clusters, processed_cells = (
                        self.layout_model.postprocess_on_page(
                            page=page, clusters=pred_clusters
                        )
                    )

                    # Define prompt structure
                    if callable(self.vlm_options.prompt):
                        user_prompt = self.vlm_options.prompt(page.parsed_page)
                    else:
                        user_prompt = self.vlm_options.prompt

                    prompt = self.formulate_prompt(user_prompt, processed_clusters)

                    generated_text, generation_time = self.vlm_model.predict_on_image(
                        page_image=page_image, prompt=prompt
                    )

                    page.predictions.vlm_response = VlmPrediction(
                        text=generated_text,
                        generation_time=generation_time,
                    )

                yield page

    def formulate_prompt(self, user_prompt: str, clusters: list[Cluster]) -> str:
        """Formulate a prompt for the VLM."""

        if self.vlm_options.transformers_prompt_style == TransformersPromptStyle.RAW:
            return user_prompt

        elif self.vlm_options.repo_id == "microsoft/Phi-4-multimodal-instruct":
            _log.debug("Using specialized prompt for Phi-4")
            # more info here: https://huggingface.co/microsoft/Phi-4-multimodal-instruct#loading-the-model-locally

            user_prompt = "<|user|>"
            assistant_prompt = "<|assistant|>"
            prompt_suffix = "<|end|>"

            prompt = f"{user_prompt}<|image_1|>{user_prompt}{prompt_suffix}{assistant_prompt}"
            _log.debug(f"prompt for {self.vlm_options.repo_id}: {prompt}")

            return prompt

        elif self.vlm_options.transformers_prompt_style == TransformersPromptStyle.CHAT:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a page from a document.",
                        },
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=False
            )
            return prompt

        raise RuntimeError(
            f"Uknown prompt style `{self.vlm_options.transformers_prompt_style}`. Valid values are {', '.join(s.value for s in TransformersPromptStyle)}."
        )
