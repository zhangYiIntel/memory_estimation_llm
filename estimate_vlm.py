import openvino as ov
import sys
import math
import json
import os

from estimate_llm import estimate_llm_memory
from estimate_vit import estimate_vit_memory

def estimate_vlm_memory(model_folder, seq_length):
    model_config_json = os.path.join(model_folder, "config.json")
    model_config = None
    with open(model_config_json, 'r') as f:
        model_config = json.load(f)
    vision_config = model_config['vision_config']
    convert_to_gb = (1024 ** 3)
    vit_const_size, vit_temp_size = estimate_vit_memory(model_folder, seq_length)
    vit_total_size = vit_const_size + vit_temp_size
    output_seq_length = seq_length / (vision_config['spatial_merge_size'] ** 2) + 360
    print("output_seq_length ", output_seq_length)
    llm_const_size, llm_temp_size = estimate_llm_memory(model_folder, output_seq_length)
    llm_total_size = llm_const_size + llm_temp_size
    vision_embedding_path = os.path.join(model_folder, "openvino_vision_embeddings_model.bin")
    text_embedding_path = os.path.join(model_folder,"openvino_text_embeddings_model.bin")
    vision_embedding_size = 0 
    if os.path.exists(vision_embedding_path):
        vision_embedding_size = os.path.getsize(vision_embedding_path)
    text_embedding_size = 0
    if os.path.exists(text_embedding_path):
        text_embedding_size = os.path.getsize(text_embedding_path)
    print(vision_embedding_size, text_embedding_size)
    print("Total VIT Runtime Memory {:.2f} GB weight {:.2f} GB temp {:.2f} GB".format(vit_total_size / convert_to_gb, vit_const_size / convert_to_gb, vit_temp_size / convert_to_gb))
    print("Total LLM Runtime Memory {:.2f} GB weight {:.2f} GB temp {:.2f} GB".format(llm_total_size / convert_to_gb, llm_const_size / convert_to_gb, llm_temp_size / convert_to_gb))
    print("Total VIT + LLM + EMBEDDING Runtime Memory {:.2f} GB".format((llm_total_size + vit_total_size + text_embedding_size + vision_embedding_size) / convert_to_gb))
    
    


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 {0} <ov model folder> <seq_length>".format(sys.argv[0]))
        exit(-1)
    model_folder = sys.argv[1]
    seq_length = int(sys.argv[2])
    estimate_vlm_memory(model_folder, seq_length)
    
    