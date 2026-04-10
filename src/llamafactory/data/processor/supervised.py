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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput

import re

logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[int], list[list], list[int], int, list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        # TODO: add token type：0=source (non-causal), 1=target (causal)
        token_types = []

        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
                # Add token type markers: 0=source (non-causal), 1=target (causal)
                token_types = [0] * len(source_ids) + [1] * len(target_ids) + token_types
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label
                # Add token type markers: 0=source (non-causal), 1=target (causal)
                token_types += [0] * len(source_ids) + [1] * len(target_ids)

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]
            token_types += [1]  # EOS token is part of target

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
        cur_label_num_ids = step_patterns['S'+str(cur_label_num)]
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
                        # if tok_i in [151652, 151653, 151655, 151656, 151645]:  # 151645: <|im_end|>
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
        assert passage_length == len(label_nums)

        has_image = [1] * len(images)
        has_video = [1] * len(videos)
        if has_image == []:
            has_image = [0] * passage_length
        if has_video == []:
            has_video = [0] * passage_length

        return input_ids, labels, token_types, range_all, ground_truth, passage_length, has_image, has_video

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels, token_types, range_all, ground_truth, passage_length, has_image, has_video = self._encode_data_example(
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
            if self.data_args.mixed_attention:
                model_inputs["token_types"].append(token_types)
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
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels, token_types = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

            # if self.data_args.mixed_attention:
            #     model_inputs["token_types"].append(token_types)

        return model_inputs
