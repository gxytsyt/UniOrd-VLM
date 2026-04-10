# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from ..extras import logging
from .data_utils import Role

import random

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .mm_plugin import AudioInput, ImageInput, VideoInput
    from .parser import DatasetAttr

    MediaType = Union[ImageInput, VideoInput, AudioInput]


logger = logging.get_logger(__name__)


@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _find_medias(self, medias: Union["MediaType", list["MediaType"], None]) -> Optional[list["MediaType"]]:
        r"""Optionally concatenate media path to media dir when loading from local disk."""
        if medias is None:
            return None
        elif not isinstance(medias, list):
            medias = [medias]
        elif len(medias) == 0:
            return None
        else:
            medias = medias[:]

        if self.dataset_attr.load_from in ["script", "file"]:
            if isinstance(medias[0], str):
                for i in range(len(medias)):
                    media_path = os.path.join(self.data_args.media_dir, medias[i])
                    if os.path.isfile(media_path):
                        medias[i] = media_path
                    else:
                        logger.warning_rank0_once(
                            f"Media {medias[i]} does not exist in `media_dir`. Use original path."
                        )
                
            elif isinstance(medias[0], list):  # for processed video frames
                # medias is a list of lists, e.g., [[frame1.jpg, frame2.jpg], [frame3.jpg, frame4.jpg]]
                for i in range(len(medias)):
                    for j in range(len(medias[i])):
                        media_path = os.path.join(self.data_args.media_dir, medias[i][j])
                        if os.path.isfile(media_path):
                            medias[i][j] = media_path
                        else:
                            logger.warning_rank0_once(
                                f"Media {medias[i][j]} does not exist in `media_dir`. Use original path."
                            )

        return medias

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""Convert a single example in the dataset to the standard format."""
        ...


@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

        if self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], str)
            and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(example[self.dataset_attr.response], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
        else:  # unsupervised
            response = []

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output



@dataclass
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen2.5-VL-7B-Instruct")

# TODO: add seqreorder dataset converter for pre-training
@dataclass
class SeqReorderDatasetConverter(DatasetConverter):
    r"""Converter for sequence reordering dataset with text and images."""

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""
        Convert sequence reordering dataset to standard format.

        Input format:
        {
            "text": ["step1", "step2", "step3", ...],
            "vis": ["image1.jpg", "image2.jpg", "image3.jpg", ...]
        }

        Output format:
        - Prompt: Interleaved images and text with indices (S0, S1, S2, ...)
        - Response: Ground truth sequence (e.g., "S0->S1->S2")
        """
        import random

        # Get text and image lists
        text_list = example.get(self.dataset_attr.text, [])
        image_list = example.get(self.dataset_attr.images, [])

        if len(text_list) > 29:
            return {
                "_prompt": [],
                "_response": [],
                "_system": example.get(self.dataset_attr.system, ""),
                "_tools": example.get(self.dataset_attr.tools, ""),
                "_images": None,
                "_videos": None,
                "_audios": None,
            }

        # Truncate text to 200 words (for English text)
        # Split by whitespace and keep first 200 words
        MAX_WORDS = 200
        truncated_text_list = []
        for text in text_list:
            if not text:
                truncated_text_list.append("")
                continue

            # Split by whitespace
            text = text.replace('<image>', 'image').replace('<video>', 'video').replace('<audio>', 'audio')
            words = text.split()

            # Truncate to MAX_WORDS
            if len(words) > MAX_WORDS:
                truncated_words = words[:MAX_WORDS]
                truncated_text = " ".join(truncated_words)
                logger.warning_rank0_once(
                    f"Text truncated from {len(words)} to {MAX_WORDS} words."
                )
            else:
                truncated_text = text
            len_tokenize = len(tokenizer.tokenize(truncated_text))

            if len_tokenize >= 750:
                return {
                    "_prompt": [],
                    "_response": [],
                    "_system": example.get(self.dataset_attr.system, ""),
                    "_tools": example.get(self.dataset_attr.tools, ""),
                    "_images": None,
                    "_videos": None,
                    "_audios": None,
                }

            truncated_text_list.append(truncated_text)

        # Replace original text_list with truncated version
        text_list = truncated_text_list

        if not text_list or not image_list:
            raise ValueError("Both 'text' and 'images' fields are required for seqreorder format.")

        if len(text_list) != len(image_list):
            raise ValueError(f"Length mismatch: text has {len(text_list)} items, images has {len(image_list)} items.")

        # Create paired list with original indices
        n = len(text_list)
        paired_data = [(i, text_list[i], image_list[i]) for i in range(n)]

        # Filter out invalid image paths before shuffling
        valid_paired_data = []
        for orig_idx, text, image in paired_data:
            # Replace /data with /workspace for Docker compatibility
            if isinstance(image, str) and image.startswith('/data/'):
                image = image.replace('/data/', '/workspace/', 1)

            # Check if image file exists
            image_path = image
            if self.dataset_attr.load_from in ["script", "file"]:
                # Try with media_dir prefix
                test_path = os.path.join(self.data_args.media_dir, image)
                if os.path.isfile(test_path):
                    image_path = image
                elif os.path.isfile(image):
                    image_path = image
                else:
                    # Image not found, skip this step
                    logger.warning_rank0_once(
                        f"Image not found: {image} (also tried {test_path}). Skipping this step."
                    )
                    continue

            valid_paired_data.append((orig_idx, text, image_path))

        if len(valid_paired_data) == 0:
            raise ValueError("No valid images found in this example. All image paths are invalid.")
        if len(valid_paired_data) == 1:
            print("len_seq == 1, no need")
            return {
                "_prompt": [],
                "_response": [],
                "_system": example.get(self.dataset_attr.system, ""),
                "_tools": example.get(self.dataset_attr.tools, ""),
                "_images": None,
                "_videos": None,
                "_audios": None,
            }

        # Shuffle the valid paired data
        shuffled_data = valid_paired_data.copy()
        random.shuffle(shuffled_data)

        # Create mapping from original index to shuffled position
        orig_to_shuffled_pos = {}
        for shuffled_pos, (orig_idx, _, _) in enumerate(shuffled_data):
            orig_to_shuffled_pos[orig_idx] = shuffled_pos

        # Ground truth: the shuffled positions in original order
        valid_orig_indices = sorted([orig_idx for orig_idx, _, _ in valid_paired_data])
        # ground_truth = "->".join([f"{orig_to_shuffled_pos[orig_idx]}" for orig_idx in valid_orig_indices])
        ground_truth = ", ".join([f"{orig_to_shuffled_pos[orig_idx]}" for orig_idx in valid_orig_indices])

        # Build interleaved prompt with shuffled order
        base_prompt = 'Given several steps, each labeled as S0, S1, ..., SN, where each step contains an image and a text description, determine their correct temporal order based on the coherence and logical progression of the content. Output only the predicted order of the labels in the format: X->Y->...->Z.'
        prompt_content = base_prompt + '\n'
        shuffled_images = []

        for idx, (orig_idx, text, image) in enumerate(shuffled_data):
            prompt_content += f"S{idx}: {text} <image>\n"
            shuffled_images.append(image)

        prompt_content = prompt_content.strip()

        messages = [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": ground_truth}
        ]

        # Log filtering results
        if len(valid_paired_data) < n:
            logger.warning_rank0_once(
                f"Filtered out {n - len(valid_paired_data)} invalid images. "
                f"Using {len(valid_paired_data)} valid images."
            )

        output = {
            "_prompt": messages[:-1],
            "_response": messages[-1:],
            "_system": example.get(self.dataset_attr.system, ""),
            "_tools": example.get(self.dataset_attr.tools, ""),
            "_images": self._find_medias(shuffled_images) if shuffled_images else None,
            "_videos": None,
            "_audios": None,
        }
        return output

@dataclass
class SeqReorderPureTextDatasetConverter(DatasetConverter):
    r"""Converter for sequence reordering dataset with text and images."""

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""
        Convert sequence reordering dataset to standard format.

        Input format:
        {
            "text": ["step1", "step2", "step3", ...],
            "vis": ["image1.jpg", "image2.jpg", "image3.jpg", ...]
        }

        Output format:
        - Prompt: Interleaved images and text with indices (S0, S1, S2, ...)
        - Response: Ground truth sequence (e.g., "S0->S1->S2")
        """
        import random

        text_list = example.get(self.dataset_attr.text, [])

        if len(text_list) > 29:
            return {
                "_prompt": [],
                "_response": [],
                "_system": example.get(self.dataset_attr.system, ""),
                "_tools": example.get(self.dataset_attr.tools, ""),
                "_images": None,
                "_videos": None,
                "_audios": None,
            }

        MAX_WORDS = 200
        truncated_text_list = []
        for text in text_list:
            if not text:
                truncated_text_list.append("")
                continue

            # Split by whitespace
            text = text.replace('<image>', 'image').replace('<video>', 'video').replace('<audio>', 'audio')
            words = text.split()

            # Truncate to MAX_WORDS
            if len(words) > MAX_WORDS:
                truncated_words = words[:MAX_WORDS]
                truncated_text = " ".join(truncated_words)
                logger.warning_rank0_once(
                    f"Text truncated from {len(words)} to {MAX_WORDS} words."
                )
            else:
                truncated_text = text
            len_tokenize = len(tokenizer.tokenize(truncated_text))

            if len_tokenize >= 750:
                return {
                    "_prompt": [],
                    "_response": [],
                    "_system": example.get(self.dataset_attr.system, ""),
                    "_tools": example.get(self.dataset_attr.tools, ""),
                    "_images": None,
                    "_videos": None,
                    "_audios": None,
                }

            truncated_text_list.append(truncated_text)

        text_list = truncated_text_list

        # Create paired list with original indices
        n = len(text_list)
        paired_data = [(i, text_list[i]) for i in range(n)]

        # Filter out invalid image paths before shuffling
        valid_paired_data = []
        for orig_idx, text in paired_data:
            valid_paired_data.append((orig_idx, text))

        if len(valid_paired_data) == 1 or len(valid_paired_data) == 0:
            print("len_seq == 1/0, no need")
            return {
                "_prompt": [],
                "_response": [],
                "_system": example.get(self.dataset_attr.system, ""),
                "_tools": example.get(self.dataset_attr.tools, ""),
                "_images": None,
                "_videos": None,
                "_audios": None,
            }

        # Shuffle the valid paired data
        shuffled_data = valid_paired_data.copy()
        random.shuffle(shuffled_data)

        # Create mapping from original index to shuffled position
        orig_to_shuffled_pos = {}
        for shuffled_pos, (orig_idx, _) in enumerate(shuffled_data):
            orig_to_shuffled_pos[orig_idx] = shuffled_pos

        # Ground truth: the shuffled positions in original order
        valid_orig_indices = sorted([orig_idx for orig_idx, _ in valid_paired_data])
        # ground_truth = "->".join([f"S{orig_to_shuffled_pos[orig_idx]}" for orig_idx in valid_orig_indices])
        ground_truth = ", ".join([f"{orig_to_shuffled_pos[orig_idx]}" for orig_idx in valid_orig_indices])

        # Build interleaved prompt with shuffled order
        base_prompt = 'Given several sentences, each labeled as S0, S1, ..., SN, where each step contains a text description, determine their correct temporal order based on the coherence and logical progression of the content. Output only the predicted order of the labels in the format: X->Y->...->Z.'
        prompt_content = base_prompt + '\n'

        for idx, (orig_idx, text) in enumerate(shuffled_data):
            prompt_content += f"S{idx}: {text}\n"

        prompt_content = prompt_content.strip()

        messages = [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": ground_truth}
        ]

        output = {
            "_prompt": messages[:-1],
            "_response": messages[-1:],
            "_system": example.get(self.dataset_attr.system, ""),
            "_tools": example.get(self.dataset_attr.tools, ""),
            "_images": None,
            "_videos": None,
            "_audios": None,
        }
        return output


@dataclass
class SeqReorderDatasetPureImageConverter(DatasetConverter):
    r"""Converter for sequence reordering dataset with text and images."""

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""
        Convert sequence reordering dataset to standard format.

        Input format:
        {
            "text": ["step1", "step2", "step3", ...],
            "vis": ["image1.jpg", "image2.jpg", "image3.jpg", ...]
        }

        Output format:
        - Prompt: Interleaved images and text with indices (S0, S1, S2, ...)
        - Response: Ground truth sequence (e.g., "S0->S1->S2")
        """
        import random

        text_list = []
        image_list = example.get(self.dataset_attr.images, [])

        # Create paired list with original indices
        n = len(image_list)
        paired_data = [(i, image_list[i]) for i in range(n)]

        # Filter out invalid image paths before shuffling
        valid_paired_data = []
        for orig_idx, image in paired_data:
            # Replace /data with /workspace for Docker compatibility
            if isinstance(image, str) and image.startswith('/data/'):
                image = image.replace('/data/', '/workspace/', 1)

            # Check if image file exists
            image_path = image
            if self.dataset_attr.load_from in ["script", "file"]:
                # Try with media_dir prefix
                test_path = os.path.join(self.data_args.media_dir, image)
                if os.path.isfile(test_path):
                    image_path = image
                elif os.path.isfile(image):
                    image_path = image
                else:
                    # Image not found, skip this step
                    logger.warning_rank0_once(
                        f"Image not found: {image} (also tried {test_path}). Skipping this step."
                    )
                    continue

            valid_paired_data.append((orig_idx, image_path))

        if len(valid_paired_data) == 0 or len(valid_paired_data) == 1:
            print("len_seq == 1, no need")
            return {
                "_prompt": [],
                "_response": [],
                "_system": example.get(self.dataset_attr.system, ""),
                "_tools": example.get(self.dataset_attr.tools, ""),
                "_images": None,
                "_videos": None,
                "_audios": None,
            }

        # Shuffle the valid paired data
        shuffled_data = valid_paired_data.copy()
        random.shuffle(shuffled_data)

        # Create mapping from original index to shuffled position
        orig_to_shuffled_pos = {}
        for shuffled_pos, (orig_idx, _) in enumerate(shuffled_data):
            orig_to_shuffled_pos[orig_idx] = shuffled_pos

        # Ground truth: the shuffled positions in original order
        valid_orig_indices = sorted([orig_idx for orig_idx, _ in valid_paired_data])
        # ground_truth = "->".join([f"S{orig_to_shuffled_pos[orig_idx]}" for orig_idx in valid_orig_indices])
        ground_truth = ", ".join([f"{orig_to_shuffled_pos[orig_idx]}" for orig_idx in valid_orig_indices])

        # Build interleaved prompt with shuffled order
        base_prompt = 'Given several steps, each labeled as S0, S1, ..., SN, where each step contains an image, determine their correct temporal order based on the coherence and logical progression of the content. Output only the predicted order of the labels in the format: X->Y->...->Z.'
        prompt_content = base_prompt + '\n'
        shuffled_images = []

        for idx, (orig_idx, image) in enumerate(shuffled_data):
            prompt_content += f"S{idx}: <image>\n"
            shuffled_images.append(image)

        prompt_content = prompt_content.strip()

        messages = [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": ground_truth}
        ]

        # Log filtering results
        if len(valid_paired_data) < n:
            logger.warning_rank0_once(
                f"Filtered out {n - len(valid_paired_data)} invalid images. "
                f"Using {len(valid_paired_data)} valid images."
            )

        output = {
            "_prompt": messages[:-1],
            "_response": messages[-1:],
            "_system": example.get(self.dataset_attr.system, ""),
            "_tools": example.get(self.dataset_attr.tools, ""),
            "_images": self._find_medias(shuffled_images) if shuffled_images else None,
            "_videos": None,
            "_audios": None,
        }
        return output


@dataclass
class SeqReorderDatasetVideoTextConverter(DatasetConverter):
    r"""Converter for sequence reordering dataset with text and images."""

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""
        Convert sequence reordering dataset to standard format.

        Input format:
        {
            "text": ["step1", "step2", "step3", ...],
            "vis": ["image1.jpg", "image2.jpg", "image3.jpg", ...]
        }

        Output format:
        - Prompt: Interleaved images and text with indices (S0, S1, S2, ...)
        - Response: Ground truth sequence (e.g., "S0->S1->S2")
        """
        import random

        # Get text and image lists
        text_list = example.get(self.dataset_attr.text, [])
        video_list = example.get(self.dataset_attr.images, [])

        # if len(text_list) > 29:
        if len(text_list) > 15:
            return {
                "_prompt": [],
                "_response": [],
                "_system": example.get(self.dataset_attr.system, ""),
                "_tools": example.get(self.dataset_attr.tools, ""),
                "_images": None,
                "_videos": None,
                "_audios": None,
            }

        # Truncate text to 200 words (for English text)
        # Split by whitespace and keep first 200 words
        MAX_WORDS = 200
        truncated_text_list = []
        for text in text_list:
            if not text:
                truncated_text_list.append("")
                continue

            # Split by whitespace
            text = text.replace('<image>', 'image').replace('<video>', 'video').replace('<audio>', 'audio')
            words = text.split()

            # Truncate to MAX_WORDS
            if len(words) > MAX_WORDS:
                truncated_words = words[:MAX_WORDS]
                truncated_text = " ".join(truncated_words)
                logger.warning_rank0_once(
                    f"Text truncated from {len(words)} to {MAX_WORDS} words."
                )
            else:
                truncated_text = text
            len_tokenize = len(tokenizer.tokenize(truncated_text))

            if len_tokenize >= 750:
                return {
                    "_prompt": [],
                    "_response": [],
                    "_system": example.get(self.dataset_attr.system, ""),
                    "_tools": example.get(self.dataset_attr.tools, ""),
                    "_images": None,
                    "_videos": None,
                    "_audios": None,
                }

            truncated_text_list.append(truncated_text)

        # Replace original text_list with truncated version
        text_list = truncated_text_list

        if not text_list or not video_list:
            raise ValueError("Both 'text' and 'images' fields are required for seqreorder format.")

        if len(text_list) != len(video_list):
            raise ValueError(f"Length mismatch: text has {len(text_list)} items, images has {len(image_list)} items.")

        # Create paired list with original indices
        n = len(text_list)
        paired_data = [(i, text_list[i], video_list[i]) for i in range(n)]

        # Filter out invalid image paths before shuffling
        valid_paired_data = []
        for orig_idx, text, video in paired_data:
            # Replace /data with /workspace for Docker compatibility
            if isinstance(video, str) and video.startswith('/data/'):
                video = video.replace('/data/', '/workspace/', 1)

            # Check if image file exists
            video_path = video
            if self.dataset_attr.load_from in ["script", "file"]:
                # Try with media_dir prefix
                test_path = os.path.join(self.data_args.media_dir, video_path)
                if os.path.isfile(test_path):
                    video_path = video
                elif os.path.isfile(video):
                    video_path = video
                else:
                    # Image not found, skip this step
                    logger.warning_rank0_once(
                        f"Image not found: {video} (also tried {test_path}). Skipping this step."
                    )
                    continue

            valid_paired_data.append((orig_idx, text, video_path))

        if len(valid_paired_data) == 0:
            raise ValueError("No valid images found in this example. All image paths are invalid.")
        if len(valid_paired_data) == 1:
            print("len_seq == 1, no need")
            return {
                "_prompt": [],
                "_response": [],
                "_system": example.get(self.dataset_attr.system, ""),
                "_tools": example.get(self.dataset_attr.tools, ""),
                "_images": None,
                "_videos": None,
                "_audios": None,
            }


        # Shuffle the valid paired data
        shuffled_data = valid_paired_data.copy()
        random.shuffle(shuffled_data)

        # Create mapping from original index to shuffled position
        orig_to_shuffled_pos = {}
        for shuffled_pos, (orig_idx, _, _) in enumerate(shuffled_data):
            orig_to_shuffled_pos[orig_idx] = shuffled_pos

        # Ground truth: the shuffled positions in original order
        valid_orig_indices = sorted([orig_idx for orig_idx, _, _ in valid_paired_data])
        # ground_truth = "->".join([f"{orig_to_shuffled_pos[orig_idx]}" for orig_idx in valid_orig_indices])
        ground_truth = ", ".join([f"{orig_to_shuffled_pos[orig_idx]}" for orig_idx in valid_orig_indices])

        # Build interleaved prompt with shuffled order
        # base_prompt = 'Given several steps, each labeled as S0, S1, ..., SN, where each step contains an image and a text description, determine their correct temporal order based on the coherence and logical progression of the content. Output only the predicted order of the labels in the format: SX->SY->...->SZ.'
        base_prompt = 'Given several steps, each labeled as S0, S1, ..., SN, where each step contains an video and a text description, determine their correct temporal order based on the coherence and logical progression of the content. Output only the predicted order of the labels in the format: X->Y->...->Z.'
        prompt_content = base_prompt + '\n'
        shuffled_videos = []

        for idx, (orig_idx, text, video) in enumerate(shuffled_data):
            prompt_content += f"S{idx}: {text} <video>\n"
            shuffled_videos.append(video)

        prompt_content = prompt_content.strip()

        messages = [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": ground_truth}
        ]

        # Log filtering results
        if len(valid_paired_data) < n:
            logger.warning_rank0_once(
                f"Filtered out {n - len(valid_paired_data)} invalid videos. "
                f"Using {len(valid_paired_data)} valid videos."
            )

        output = {
            "_prompt": messages[:-1],
            "_response": messages[-1:],
            "_system": example.get(self.dataset_attr.system, ""),
            "_tools": example.get(self.dataset_attr.tools, ""),
            "_images": None,
            "_videos": self._find_medias(shuffled_videos) if shuffled_videos else None,
            "_audios": None,
        }
        return output


@dataclass
class SeqReorderDatasetPureVideoConverter(DatasetConverter):
    r"""Converter for sequence reordering dataset with text and images."""

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:

        import random

        text_list = []
        video_list = example.get(self.dataset_attr.images, [])

        # Create paired list with original indices
        n = len(video_list)
        paired_data = [(i, video_list[i]) for i in range(n)]

        # Filter out invalid image paths before shuffling
        valid_paired_data = []
        for orig_idx, video in paired_data:
            # Replace /data with /workspace for Docker compatibility
            if isinstance(video, str) and video.startswith('/data/'):
                video = video.replace('/data/', '/workspace/', 1)

            # Check if image file exists
            video_path = video
            if self.dataset_attr.load_from in ["script", "file"]:
                # Try with media_dir prefix
                test_path = os.path.join(self.data_args.media_dir, video_path)
                if os.path.isfile(test_path):
                    video_path = video
                elif os.path.isfile(video):
                    video_path = video
                else:
                    # Image not found, skip this step
                    logger.warning_rank0_once(
                        f"Image not found: {video} (also tried {test_path}). Skipping this step."
                    )
                    continue

            valid_paired_data.append((orig_idx, video_path))

        if len(valid_paired_data) == 0 or len(valid_paired_data) == 1:
            print("len_seq == 0/1, no need")
            return {
                "_prompt": [],
                "_response": [],
                "_system": example.get(self.dataset_attr.system, ""),
                "_tools": example.get(self.dataset_attr.tools, ""),
                "_images": None,
                "_videos": None,
                "_audios": None,
            }

        # Shuffle the valid paired data
        shuffled_data = valid_paired_data.copy()
        random.shuffle(shuffled_data)

        # Create mapping from original index to shuffled position
        orig_to_shuffled_pos = {}
        for shuffled_pos, (orig_idx, _) in enumerate(shuffled_data):
            orig_to_shuffled_pos[orig_idx] = shuffled_pos

        # Ground truth: the shuffled positions in original order
        valid_orig_indices = sorted([orig_idx for orig_idx, _ in valid_paired_data])
        # ground_truth = "->".join([f"{orig_to_shuffled_pos[orig_idx]}" for orig_idx in valid_orig_indices])
        ground_truth = ", ".join([f"{orig_to_shuffled_pos[orig_idx]}" for orig_idx in valid_orig_indices])

        # Build interleaved prompt with shuffled order
        # base_prompt = 'Given several steps, each labeled as S0, S1, ..., SN, where each step contains an image and a text description, determine their correct temporal order based on the coherence and logical progression of the content. Output only the predicted order of the labels in the format: SX->SY->...->SZ.'
        base_prompt = 'Given several steps, each labeled as S0, S1, ..., SN, where each step contains an video, determine their correct temporal order based on the coherence and logical progression of the content. Output only the predicted order of the labels in the format: X->Y->...->Z.'
        prompt_content = base_prompt + '\n'
        shuffled_videos = []

        for idx, (orig_idx, video) in enumerate(shuffled_data):
            prompt_content += f"S{idx}: <video>\n"
            shuffled_videos.append(video)

        prompt_content = prompt_content.strip()

        messages = [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": ground_truth}
        ]

        # Log filtering results
        if len(valid_paired_data) < n:
            logger.warning_rank0_once(
                f"Filtered out {n - len(valid_paired_data)} invalid videos. "
                f"Using {len(valid_paired_data)} valid videos."
            )

        output = {
            "_prompt": messages[:-1],
            "_response": messages[-1:],
            "_system": example.get(self.dataset_attr.system, ""),
            "_tools": example.get(self.dataset_attr.tools, ""),
            "_images": None,
            "_videos": self._find_medias(shuffled_videos) if shuffled_videos else None,
            "_audios": None,
        }
        return output


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,

    # TODO: add seqreorder dataset converter for pre-training
    "seqreorder": SeqReorderDatasetConverter,
    "seqreorder_t": SeqReorderPureTextDatasetConverter,
    "seqreorder_i": SeqReorderDatasetPureImageConverter,
    "seqreorder_vt": SeqReorderDatasetVideoTextConverter,
    "seqreorder_v": SeqReorderDatasetPureVideoConverter,

}


def register_dataset_converter(name: str, dataset_converter: type["DatasetConverter"]) -> None:
    r"""Register a new dataset converter."""
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")

    DATASET_CONVERTERS[name] = dataset_converter


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments") -> "DatasetConverter":
    r"""Get a dataset converter."""
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Align the dataset to a specific format.

    Aligned dataset:
    _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
    _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
    _system: "..."
    _tools: "..."
    _images: []
    _videos: []
    _audios: []
    """
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    return dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )
