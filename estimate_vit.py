import openvino as ov
import sys
import math
import json
import os

def estimate_vit_memory(model_folder, seq_length):
    vision_embedding_path = os.path.join(model_folder, "openvino_vision_embeddings_model.bin")
    vision_embedding_size = 0 
    # check vision_embedding model
    if os.path.exists(vision_embedding_path):
        vision_embedding_size = os.path.getsize(vision_embedding_path)
    merger_path = os.path.join(model_folder, "openvino_vision_embeddings_merger_model.bin")
    # check mermger model
    vision_merger_size = 0
    if os.path.exists(merger_path):
        vision_merger_size = os.path.getsize(merger_path)

    # check resampler model
    resampler_size = 0
    resampler_path = os.path.join(model_folder, "openvino_resampler_model.bin")
    if os.path.exists(resampler_path):
        resampler_size = os.path.getsize(resampler_path)
    
    projection_size = 0
    vision_proj_path = os.path.join(model_folder, "openvino_vision_projection_model.bin")
    if os.path.exists(vision_proj_path):
        projection_size = os.path.getsize(vision_proj_path)
    model_config_json = os.path.join(model_folder, "config.json")

    model_config = None
    with open(model_config_json, 'r') as f:
        model_config = json.load(f)

    # model.reshape({"input_ids": [-1, 1], "attention_mask": [-1, 1], "position_ids": [-1, 1], "beam_idx": [-1] })
    const_size = vision_embedding_size + vision_merger_size + resampler_size + projection_size

    attn_compenents = ['embed_tokens', 'input_layernorm', 'q_proj', 'k_proj', 'v_proj', 'scaled_dot_product_attention', 'mlp','post_attention_layernorm''merger']
    vision_config = model_config['vision_config'] if "vision_config" in model_config else model_config
    if model_config['model_type'] == "phi3_v":
        # Phi-3 uses hardcoded CLIP VIT
        num_attention_heads = 16
        hidden_size = 1024
        intermediate_size = 4096
        kv_hidden_size = hidden_size
        head_dim = hidden_size // num_attention_heads
    else:   
        num_attention_heads = vision_config['num_heads'] if "num_heads" in vision_config else vision_config['num_attention_heads']
        out_hidden_size= vision_config['out_hidden_size'] if "out_hidden_size" in vision_config else vision_config["hidden_size"]
        hidden_size = vision_config['hidden_size']
        # vit doesn't have multi-query
        kv_hidden_size = hidden_size
        head_dim = vision_config['head_dim'] if 'head_dim' in vision_config else hidden_size // num_attention_heads
        intermediate_size = vision_config['intermediate_size']
    model_type = None
    has_rope = False
    if 'model_type' in vision_config and vision_config['model_type'] == "qwen2_5_vl":
        has_rope = True
        attn_compenents.append('merger')
    if 'architectures' in vision_config:
        model_type = "InternVisionModel"
    if model_type == "InternVisionModel":
        has_rope = False
    component_size = {
        'q_proj': hidden_size,
        'k_proj': kv_hidden_size,
        'v_proj': kv_hidden_size,
        'hidden_states_input': hidden_size,
        'input_layernorm': 0,
        'post_attention_layernorm': 0,
        'scaled_dot_product_attention': 0
    }
    
    if has_rope:
        component_size['rotary_emb'] = True

    if 'intermediate_size' in vision_config:
        component_size['mlp'] = intermediate_size

    if 'spatial_merge_size' in vision_config:
        component_size['merger'] = seq_length // (vision_config['spatial_merge_size'] ** 2)

    if resampler_size:
        attn_compenents.append("resampler")
        
        resampler_k_proj = model_config["hidden_size"]
        # for query of resampler in minicpm, the length is fixed as batch * model_config["query_num"] * model_config["hidden_size"]
        resampler_q_proj = 10 * model_config["query_num"] * model_config["hidden_size"]
        resampler_v_proj = model_config["hidden_size"]
        component_size['resampler'] = (resampler_q_proj + resampler_k_proj + resampler_v_proj) * 2
    
    if projection_size:
        attn_compenents.append("projection")
        mlp1 = model_config["hidden_size"]
        mlp2 = model_config["hidden_size"]
        # hardcode the image length to 757 here
        component_size['projection'] = (mlp1 + mlp2) * 2 * 757

    print(component_size)
    temp_size = 0   
    element_size = 2
    # calculate internal buffer inside hidden layer, assume that the internal memory could be shared between different layers
    for comp_name in attn_compenents:
        if comp_name == 'hidden_states':
            added_size = seq_length * component_size[comp_name] * element_size
            print("Add Embed tokens prev {0} cure {1}".format(temp_size, added_size))
            temp_size = temp_size + added_size
        if comp_name in ['q_proj', 'k_proj', 'v_proj']:
            added_size = seq_length * component_size[comp_name] * element_size
            print("Add {0} prev {1} cure {2}".format(comp_name, temp_size, added_size))
            temp_size = temp_size + added_size
        if comp_name in ['input_layernorm']:
            # 'post_attention_layernorm' could reuse the memory from input_layernorm
            added_size = seq_length * component_size[comp_name] * element_size
            print("Add {0} prev {1} cure {2}".format(comp_name, temp_size, added_size))
            temp_size = temp_size + added_size
        if comp_name == 'mlp':
            # o_proj/rms/down_proj should reuse qkv_proj
            # only  up_proj/gate_proj needs extra buffer
            added_size = (component_size[comp_name] * 2) * seq_length * element_size
            print("Add {0} prev {1} cure {2}".format(comp_name, temp_size, added_size))
            temp_size = temp_size + added_size
            # temp_size = (3 * hidden_size + component_size[comp_name] * 2) * seq_length * element_size + temp_size
        if comp_name == 'rotary_emb':
            # assume that cos/sin table rope subgraph all runs in F32
            # rotary_pos_emb
            print("Add rotary_emb")
            temp_size = temp_size + seq_length * head_dim // 2 * 4
            # concat
            temp_size = temp_size + seq_length * head_dim * 4
            # cos_table
            temp_size = temp_size + seq_length * head_dim * 4
            # sin_table
            temp_size = temp_size + seq_length * head_dim * 4
            
            # rope not inlined
            # rope k
            temp_size = temp_size + seq_length * component_size['k_proj'] * 2
            # rope q
            temp_size = temp_size + seq_length * component_size['q_proj'] * 2
        if comp_name == 'merger':
            print("Add merger")
            mlp2 =  component_size['merger'] * out_hidden_size
            # first FC could resuse RMS input
            # allocate output for 2nd FC
            temp_size = temp_size + mlp2
        if comp_name == 'resampler':
            print("Add resampler")
            temp_size = temp_size + component_size['resampler']
        if comp_name == 'projection':
            print("Add projection")
            temp_size = temp_size + component_size['projection']
            
    # print(temp_size)
    # input_ids
    # position_ids
    input_size = seq_length * 4 * hidden_size
    # window attention_mask is a special input of qwen2_5
    window_attention_mask_size = seq_length * seq_length if vision_config['model_type'] == "qwen2_5_vl" else 0
    temp_size = temp_size + window_attention_mask_size + input_size
    return const_size, temp_size


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 {0} <ov model folder> <seq_length>".format(sys.argv[0]))
        exit(-1)
    model_folder = sys.argv[1]
    seq_length = int(sys.argv[2])
    const_size, temp_size = estimate_vit_memory(model_folder, seq_length)
    total_size = const_size + temp_size
    convert_to_gb = (1024 ** 3)
    print("Total Runtime Memory {:.2f} GB weight {:.2f} GB temp {:.2f} GB".format(total_size / convert_to_gb, const_size / convert_to_gb, temp_size / convert_to_gb))