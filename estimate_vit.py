import openvino as ov
import sys
import math
import json
import os


if len(sys.argv) < 3:
    print("Usage: python3 {0} <ov model folder> <seq_length>".format(sys.argv[0]))
    exit(-1)
model_folder = sys.argv[1]
model_path = os.path.join(model_folder, "openvino_vision_embeddings_merger_model.xml")
if not os.path.exists(model_path):
    print("Please check path model_path")
    exit(-1)

core = ov.Core()
model = core.read_model(model_path)
model_config_json = os.path.join(model_folder, "config.json")

model_config = None
with open(model_config_json, 'r') as f:
    model_config = json.load(f)

# model.reshape({"input_ids": [-1, 1], "attention_mask": [-1, 1], "position_ids": [-1, 1], "beam_idx": [-1] })

ops_in_topo = model.get_ordered_ops()
const_size = 0
id = 0
seq_length = int(sys.argv[2])

attn_compenents = ['embed_tokens', 'input_layernorm', 'q_proj', 'k_proj', 'v_proj', 'rotary_emb', 'scaled_dot_product_attention', 'mlp','post_attention_layernorm', 'merger']
vision_config = model_config['vision_config']
num_attention_heads = vision_config['num_heads']
spatial_merge_size = vision_config['spatial_merge_size']
out_hidden_size= vision_config['out_hidden_size']

hidden_size = vision_config['hidden_size']
# vit doesn't have multi-query
kv_hidden_size = hidden_size
depth = vision_config['depth']
head_dim = vision_config['head_dim'] if 'head_dim' in vision_config else hidden_size // num_attention_heads
window_size =  vision_config['window_size']
component_size = {
    'q_proj': hidden_size,
    'k_proj': kv_hidden_size,
    'v_proj': kv_hidden_size,
    'hidden_states_input': hidden_size,
    'rotary_emb': 0,
    'input_layernorm': 0,
    'post_attention_layernorm': 0,
    'scaled_dot_product_attention': 0
}

if 'intermediate_size' in vision_config:
    component_size['mlp'] = model_config['intermediate_size']

if 'spatial_merge_size' in vision_config:
    component_size['merger'] = seq_length // (vision_config['spatial_merge_size'] ** 2)
    
for op in ops_in_topo:
    if op.type_info.name == 'Constant':
        const_size = const_size + math.prod(op.get_shape()) * op.get_output_element_type(0).get_size()



total_size = const_size

element_size = 2
print(total_size)
# calculate internal buffer inside hidden layer, assume that the internal memory could be shared between different layers
for comp_name in attn_compenents:
    if comp_name == 'hidden_states':
        added_size = seq_length * component_size[comp_name] * element_size
        print("Add Embed tokens prev {0} cure {1}".format(total_size, added_size))
        total_size = total_size + added_size
    if comp_name in ['q_proj', 'k_proj', 'v_proj']:
        added_size = seq_length * component_size[comp_name] * element_size
        print("Add {0} prev {1} cure {2}".format(comp_name, total_size, added_size))
        total_size = total_size + added_size
    if comp_name in ['input_layernorm']:
        # 'post_attention_layernorm' could reuse the memory from input_layernorm
        added_size = seq_length * component_size[comp_name] * element_size
        print("Add {0} prev {1} cure {2}".format(comp_name, total_size, added_size))
        total_size = total_size + added_size
    if comp_name == 'mlp':
        # o_proj/rms/down_proj should reuse qkv_proj
        # only  up_proj/gate_proj needs extra buffer
        added_size = (component_size[comp_name] * 2) * seq_length * element_size
        print("Add {0} prev {1} cure {2}".format(comp_name, total_size, added_size))
        total_size = total_size + added_size
        # total_size = (3 * hidden_size + component_size[comp_name] * 2) * seq_length * element_size + total_size
    if comp_name == 'rotary_emb':
        # assume that cos/sin table rope subgraph all runs in F32
        # rotary_pos_emb 
        total_size = total_size + seq_length * head_dim // 2 * 4
        # concat
        total_size = total_size + seq_length * head_dim * 4
        # cos_table
        total_size = total_size + seq_length * head_dim * 4
        # sin_table
        total_size = total_size + seq_length * head_dim * 4
        
        # rope not inlined
        # rope k
        total_size = total_size + seq_length * component_size['k_proj'] * 2
        # rope q
        total_size = total_size + seq_length * component_size['q_proj'] * 2
    if comp_name == 'merger':
        mlp2 =  component_size['merger'] * out_hidden_size
        # first FC could resuse RMS input
        print("Add mlp2 ", mlp2)
        # allocate output for 2nd FC
        total_size = total_size + mlp2
        
# print(total_size)
# input_ids
# position_ids
input_size = seq_length * 4 * hidden_size
window_attention_mask_size = seq_length * seq_length
total_size = total_size + window_attention_mask_size + input_size
print("Total runtime memory for VIT {:.2f} GB".format(total_size / 1024 / 1024 / 1024))