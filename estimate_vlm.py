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
    vision_config = model_config['vision_config'] if "vision_config" in model_config else model_config
    convert_to_mb = (1024 ** 2)
    vit_const_size, vit_temp_size = estimate_vit_memory(model_folder, seq_length)
    vit_total_size = vit_const_size + vit_temp_size
    output_seq_length = 0
    if 'spatial_merge_size' in vision_config:
        output_seq_length = seq_length / (vision_config['spatial_merge_size'] ** 2)
    model_arch = model_config["architectures"][0]
    if model_arch == "InternVLChatModel":
        #output_seq_length = int((seq_length / 13 - 1) / 4 * 13)
        output_seq_length = 1828
    if model_arch == "MiniCPMV":
        # input_ids for llm
        output_seq_length = 359
    if model_arch == "Gemma3ForConditionalGeneration":
        output_seq_length = 272
    if model_arch == "LlavaForConditionalGeneration":
        output_seq_length = 592
    if model_arch == "Phi3VForCausalLM":
        output_seq_length = 773

    llm_const_size, llm_temp_size = estimate_llm_memory(model_folder, output_seq_length)
    llm_total_size = llm_const_size + llm_temp_size
    text_embedding_path = os.path.join(model_folder,"openvino_text_embeddings_model.bin")
    vision_embedding_size = 0
    text_embedding_size = 0
    if os.path.exists(text_embedding_path):
        text_embedding_size = os.path.getsize(text_embedding_path)
    print(vision_embedding_size, text_embedding_size)
    print("Total VIT Runtime Memory {:.2f} MB weight {:.2f} MB temp {:.2f} MB".format(vit_total_size / convert_to_mb, vit_const_size / convert_to_mb, vit_temp_size / convert_to_mb))
    print("Total LLM Runtime Memory {:.2f} MB weight {:.2f} MB temp {:.2f} MB".format(llm_total_size / convert_to_mb, llm_const_size / convert_to_mb, llm_temp_size / convert_to_mb))
    print("Total VIT + LLM + Embeddings Runtime Memory {:.2f} MB".format((llm_total_size + vit_total_size + text_embedding_size + vision_embedding_size) / convert_to_mb))
    
    


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 {0} <ov model folder> <seq_length>".format(sys.argv[0]))
        exit(-1)
    model_folder = sys.argv[1]
    seq_length = int(sys.argv[2])
    estimate_vlm_memory(model_folder, seq_length)
    
    