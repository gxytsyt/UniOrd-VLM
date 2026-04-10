# [UniOrd-VLM] - Official Implementation

The official implementation of the paper "**[UniOrd-VLM: A Structure-Aware Unified Framework for Multimodal
Sequence Ordering]**".

This codebase builds on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [Transformers](https://github.com/huggingface/transformers).

---

## 1. Environment Setup

We use the official LLaMA-Factory Docker image hiyouga/llamafactory:0.9.3:

```bash
docker pull hiyouga/llamafactory:0.9.3
docker run -it hiyouga/llamafactory:0.9.3
```
 
Optionally, install `tensorboardX` for training logging:
```bash
pip install tensorboardX
```
---

## 2. Data Preparation

**Step 1.** Download the training data from [Google Drive](https://drive.google.com/file/d/1SdL8a9Xxnag6tfWiCYFCBN4Hc2eMLfwo/view?usp=drive_link). The archive contains the text portions of all datasets along with image/video paths. Place the downloaded folder under `data/`:

```
UniOrd-VLM/
  └── data/
       └── my_data/
```

**Step 2 (optional).** If you need vision-related datasets:

- **WikiHVO**: Download the corresponding videos from [Google Drive](https://drive.google.com/file/d/1BsOQR4jvjbYD-GxVj7GwKCmlLr8uqFyx/view?usp=drive_link) and place the raw video files according to the paths referenced in the data.

- **RecipeQA & WikiHow**: The original text and image files are provided by the paper *"Understanding Multimodal Procedural Knowledge by Sequencing Multimodal Instructional Manuals"*. Please refer to that work to obtain these assets.

---

## 3. Transformers Setup
We modified the modeling files under `transformers/models/qwen2_5_vl/` to support our proposed method.
Extract the zip files inside the `transformers/` directory:

```bash
cd transformers
cat tsfm.z01 tsfm.z02 tsfm.zip >combined.zip
unzip combined.zip
```

---

## 4. Training and Evaluation

Before running, modify the following variables in `examples/train_my/train_my_vlm.sh`:
 
- `MODEL_PATH`: path to your downloaded Qwen2.5-VL model
- `PYTHONPATH`: path to the project root directory
- `OUTPUT_DIR`: path where the trained model checkpoints will be saved

**Training:**

```bash
bash examples/train_my/train_my_vlm.sh
```

Similarly, modify the following variables in `examples/test_my/predict_my_vlm.sh` before testing:
 
- `LORA_BASE_DIR`: path to the saved model directory from training
- `CHECKPOINTS`: the step number of the checkpoint to evaluate

**Evaluation:**

```bash
bash examples/test_my/predict_my_vlm.sh
```
