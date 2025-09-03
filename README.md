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
```
## 3. Estimation Assumption
* Runtime buffers are resued among different atttention blocks
* Rope is not inplace since only specific ROPE such as Rotate-half could be done inplace.
* input/output of models are all placed in devices