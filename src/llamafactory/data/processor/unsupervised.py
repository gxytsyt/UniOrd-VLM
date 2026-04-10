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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ..data_utils import Role
from .processor_utils import DatasetProcessor, infer_seqlen
import re

if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class UnsupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    # ) -> tuple[list[int], list[int]]:
    ) -> tuple[list[int], list[int], list[int], list[list], list[int], int, list[int], list[int]]:
        if len(response) == 1:
            messages = prompt + response
        else:
            messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

        messages = self.template.mm_plugin.process_messages(messages, images, videos, audios, self.processor)
        input_ids, labels = self.template.encode_oneturn(self.tokenizer, messages, system, tools)
        if self.template.efficient_eos:
            labels += [self.tokenizer.eos_token_id]

        input_ids, _ = self.template.mm_plugin.process_token_ids(
            input_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        source_len, target_len = infer_seqlen(len(input_ids), len(labels), self.data_args.cutoff_len)
        input_ids = input_ids[:source_len]
        labels = labels[:target_len]

        token_types = [0] * source_len

        step_patterns = {
            "S0": [50, 15, 25],
            "S1": [50, 16, 25],
            "S2": [50, 17, 25],
            "S3": [50, 18, 25],
            "S4": [50, 19, 25],
            "S5": [50, 20, 25],
            "S6": [50, 21, 25],
            "S7": [50, 22, 25],
            "S8": [50, 23, 25],
            "S9": [50, 24, 25],
            "S10": [50, 16, 15, 25],
            "S11": [50, 16, 16, 25],
            "S12": [50, 16, 17, 25],
            "S13": [50, 16, 18, 25],
            "S14": [50, 16, 19, 25],
            "S15": [50, 16, 20, 25],
            "S16": [50, 16, 21, 25],
            "S17": [50, 16, 22, 25],
            "S18": [50, 16, 23, 25],
            "S19": [50, 16, 24, 25],
            "S20": [50, 17, 15, 25],
            "S21": [50, 17, 16, 25],
            "S22": [50, 17, 17, 25],
            "S23": [50, 17, 18, 25],
            "S24": [50, 17, 19, 25],
            "S25": [50, 17, 20, 25],
            "S26": [50, 17, 21, 25],
            "S27": [50, 17, 22, 25],
            "S28": [50, 17, 23, 25],
            "S29": [50, 17, 24, 25],
        }
        vision_tokens = [151652, 151653, 151655]

        labels_pure = []
        for label_i in labels:
            if label_i != -100:
                labels_pure.append(label_i)
        out_label = self.tokenizer.decode(labels_pure)

        # label_nums = re.findall(r"S(\d+)", out_label)
        label_nums = re.findall(r"\d+", out_label)

        label_nums = [int(str_num) for str_num in label_nums]
        ground_truth = label_nums

        cur_label_num = 0
        cur_label_num_ids = step_patterns['S' + str(cur_label_num)]
        range_all = []
        range_all_text = []
        id_i = 0
        len_seq = len(input_ids)

        while id_i < len_seq:
            tok_i = input_ids[id_i]
            if tok_i == 50:
                if id_i + len(cur_label_num_ids) < len(input_ids) and input_ids[id_i: id_i + len(
                        cur_label_num_ids)] == cur_label_num_ids:
                    start_cur = id_i
                    start_cur_text = id_i + len(cur_label_num_ids)
                    # Get the next step pattern (steps are always consecutive)
                    next_label_num = cur_label_num + 1
                    next_label_num_ids = step_patterns.get('S' + str(next_label_num), None)
                    while id_i < len_seq:
                        tok_i = input_ids[id_i]
                        # Check if we encounter a vision token
                        # if tok_i in [151652, 151653, 151655, 151645]:  # 151645: <|im_end|>
                        if tok_i in [151645]:
                            break
                        # Check if we encounter the next step pattern
                        if tok_i == 50 and next_label_num_ids is not None:
                            if id_i + len(next_label_num_ids) <= len(input_ids) and input_ids[id_i: id_i + len(
                                    next_label_num_ids)] == next_label_num_ids:
                                # Found the next step pattern, break outer while loop
                                break
                        id_i += 1
                    end_cur = id_i
                    end_cur_text = id_i
                    range_all.append([start_cur, end_cur])
                    range_all_text.append([start_cur_text, end_cur_text])
                    cur_label_num += 1
                    cur_label_num_ids = step_patterns['S' + str(cur_label_num)]
                else:
                    id_i += 1
            else:
                id_i += 1

        passage_length = len(range_all)

        has_image = [1] * len(images)
        has_video = [1] * len(videos)
        if has_image == []:
            has_image = [0] * passage_length
        if has_video == []:
            has_video = [0] * passage_length

        range_all = [[0, 0]]

        return input_ids, labels, token_types, range_all, ground_truth, passage_length, has_image, has_video
        # return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels, token_types, range_all, ground_truth, passage_length, has_image, has_video = self._encode_data_example(
            # input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

            model_inputs["range_all"].append(range_all)
            model_inputs["ground_truth"].append(ground_truth)
            model_inputs["passage_length"].append(passage_length)
            model_inputs["has_image"].append(has_image)
            model_inputs["has_video"].append(has_video)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(self.tokenizer.decode(example["labels"], skip_special_tokens=False)))
