# Memory Estimation Usage
## 1. Estimate LLM model
```bash
python3 estimate_llm.py <ov model folder> <seq_length>
# Example output
Total runtime memory 3.14 GB
```
## 2. Estimate VIT model
```bash
python3 estimate_vit.py <ov model folder> <seq_length>
# Example output
Total VIT Runtime Memory 1.72 GB weight 1.24 GB temp 0.48 GB
```
## 3. Estimate VLM pipline
### 3.0 Input Image
W x H 1024 * 682
### 3.1 Estimate qwen2_5-3b-vl
```bash
python3 estimate_vlm.py <ov model folder> <seq_length>
# Example output
python3 estimate_vlm.py ov-qwen2_5-3b-vl 7252
Total VIT Runtime Memory 1.72 GB weight 1.24 GB temp 0.48 GB
Total LLM Runtime Memory 6.02 GB weight 5.75 GB temp 0.27 GB
Total VIT + LLM + Embeddings Runtime Memory 8.32 GB
# OV 13GB
# Llama.cpp 9340 MB
```
### 3.2 Estimate internvl2_5-4b
```bash
# batch 7 * 1025
python3 ./estimate_vlm.py /mnt/llm_irs/zhangyi7/ov-internvl2_5-4b/ 7175
Total VIT Runtime Memory 1.48 GB weight 1.25 GB temp 0.23 GB
Total LLM Runtime Memory 5.87 GB weight 5.75 GB temp 0.13 GB
Total VIT + LLM + Embeddings Runtime Memory 7.22 GB
# OV 7500 MB
# Llama.cpp 7778 MB
```
### 3.3 Estimate minicpm-v-2-5-4bit
```bash
# patch_nums 5, size 1014
python3 ./estimate_vlm.py /mnt/llm_irs/zhangyi7/ov-minicpmv/OV_FP16-4BIT_DEFAULT/ 5070
Total VIT Runtime Memory 0.59 GB weight 0.45 GB temp 0.14 GB
Total LLM Runtime Memory 3.71 GB weight 3.66 GB temp 0.05 GB
Total VIT + LLM + Embeddings Runtime Memory 4.81 GB
# OV 5400 MB
# Llama.cpp 6636 MB
```
## 3. Estimation Assumption
* Runtime buffers are resued among different atttention blocks
* Rope is not inplace since only specific ROPE such as Rotate-half could be inplace.
* input/output of models are all placed in devices