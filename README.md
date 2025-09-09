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
### 3.0 Input Data
- W x H: `1024 * 682`
- Prompt: `Please Describe this picture`
### 3.1 Estimate qwen2_5-3b-vl
```bash
python3 estimate_vlm.py <ov model folder> <seq_length>
# Example output
python3 estimate_vlm.py ov-qwen2_5-3b-vl 7252
Total VIT Runtime Memory 1.48 GB weight 1.25 GB temp 0.23 GB
Total LLM Runtime Memory 5.89 GB weight 5.75 GB temp 0.14 GB
Total VIT + LLM + Embeddings Runtime Memory 7.95 GB
# OV 13GB
# Llama.cpp 9234 MB
```
### 3.2 Estimate internvl2_5-4b
```bash
# slices 7  output_channel 1024 class_channel 1 7175 = 7 * 1025
python3 ./estimate_vlm.py /mnt/llm_irs/zhangyi7/ov-internvl2_5-4b/ 7175
Total VIT Runtime Memory 1.48 GB weight 1.25 GB temp 0.23 GB
Total LLM Runtime Memory 5.87 GB weight 5.75 GB temp 0.13 GB
Total VIT + LLM + Embeddings Runtime Memory 7.24 GB
# OV 7500 MB
# Llama.cpp 7672 MB
```
### 3.3 Estimate minicpm-v-2-5-4bit
```bash
# patch_nums 5, size 1014
python3 ./estimate_vlm.py /mnt/llm_irs/zhangyi7/ov-minicpmv/OV_FP16-4BIT_DEFAULT/ 5070
Total VIT Runtime Memory 0.59 GB weight 0.45 GB temp 0.14 GB
Total LLM Runtime Memory 3.71 GB weight 3.66 GB temp 0.05 GB
Total VIT + LLM + Embeddings Runtime Memory 4.82 GB
# OV 5400 MB
# Llama.cpp 6530 MB
```
### 3.4 Estimate gemma-3-4b-it-f16
```bash
python3 ./estimate_vlm.py gemma-3-4b-it-f16 4096
Total VIT Runtime Memory 0.89 GB weight 0.78 GB temp 0.11 GB
Total LLM Runtime Memory 7.28 GB weight 7.23 GB temp 0.05 GB
Total VIT + LLM + Embeddings Runtime Memory 9.42 GB
#OV 9800 MB
#Llama.cpp 10914MiB
```
### 3.5 Estimate llava-1.5-7b-hf-4bit
```bash
Total VIT Runtime Memory 0.31 GB weight 0.29 GB temp 0.01 GB
Total LLM Runtime Memory 3.43 GB weight 3.26 GB temp 0.17 GB
Total VIT + LLM + Embeddings Runtime Memory 3.87 GB
#OV ~ 4500 MB
#Llama.cpp 7392 MB
```
## 4. Estimation Assumption
* Runtime buffers are reused among different atttention blocks, ellipse refers to output memory of each node, while ellipses refers to resued buffer, blue ellipse refers to allocated buffer.
![image info](./memory_analysis_attn.png)
* Temp buffer here refers to output memory of each Op
* Rope is not inplace since only specific ROPE such as Rotate-half could be inplace.
* input/output of models are all placed in devices
* After images are embedded by Conv2D, the only memory used is the embedded image size which is `[seq_length, hidden_size]`