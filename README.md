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
```bash
python3 estimate_vlm.py <ov model folder> <seq_length>
# Example output
python3 estimate_vlm.py ov-qwen2_5-3b-vl 7252
Total VIT Runtime Memory 1.72 GB weight 1.24 GB temp 0.48 GB
Total LLM Runtime Memory 6.02 GB weight 5.75 GB temp 0.27 GB
Total VIT + LLM + EMBEDDING Runtime Memory 8.32 GB
```
## 3. Estimation Assumption
* Runtime buffers are resued among different atttention blocks
* Rope is not inplace since only specific ROPE such as Rotate-half could be inplace.
* input/output of models are all placed in devices