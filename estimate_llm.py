import openvino as ov
import sys
import math
import json
import os


if len(sys.argv) < 3:
    print("Usage: python3 {0} <ov model folder> <seq_length>".format(sys.argv[0]))
    exit(-1)
model_folder = sys.argv[1]
model_path = os.path.join(model_folder, "openvino_language_model.xml")
if not os.path.exists(model_path):
    model_path = os.path.join(model_folder, "openvino_model.xml")
core = ov.Core()
model = core.read_model(model_path)
model_config_json = os.path.join(model_folder, "config.json");

model_config = None
with open(model_config_json, 'r') as f:
    model_config = json.load(f)

ops_in_topo = model.get_ordered_ops()
const_size = 0
attn_compenents = ['embed_tokens', 'input_layernorm', 'q_proj', 'k_proj', 'v_proj', 'rotary_emb', 'scaled_dot_product_attention', 'mlp','post_attention_layernorm']
# vlm model 
text_config = model_config['text_config'] if 'text_config' in model_config else model_config
num_attention_heads = text_config['num_attention_heads']
num_key_value_heads = text_config['num_key_value_heads']

hidden_size = text_config['hidden_size']
kv_group_num = num_attention_heads / num_key_value_heads
kv_hidden_size = int(text_config['hidden_size'] / kv_group_num)
num_hidden_layers = text_config['num_hidden_layers']
head_dim = text_config['head_dim'] if 'head_dim' in text_config else hidden_size // num_attention_heads
vocab_size =  text_config['vocab_size']
component_size = {
    'q_proj': hidden_size,
    'k_proj': kv_hidden_size,
    'v_proj': kv_hidden_size,
    'embed_tokens': hidden_size,
    'rotary_emb': 0,
    # 'input_layernorm': 0,
    # 'post_attention_layernorm': 0,
    'scaled_dot_product_attention': 0
}
if 'rms_norm_eps' in model_config:
    component_size['input_layernorm'] = hidden_size
    component_size['post_attention_layernorm'] = hidden_size

if 'intermediate_size' in model_config:
    component_size['mlp'] = model_config['intermediate_size']
    
for op in ops_in_topo:
    if op.type_info.name == 'Constant':
        const_size = const_size + math.prod(op.get_shape()) * op.get_output_element_type(0).get_size()

total_size = const_size
seq_length = 3577 + 360
element_size = 2
print("const size ", total_size)
# calculate internal buffer inside hidden layer, assume that the internal memory could be shared between different layers
for comp_name in attn_compenents:
    if comp_name == 'embed_tokens':
        added_size = seq_length * component_size[comp_name] * element_size
        print("Add Embed tokens prev {0} cur {1}".format(total_size, added_size))
        total_size = total_size + added_size
    if comp_name in ['q_proj', 'k_proj', 'v_proj']:
        added_size = seq_length * component_size[comp_name] * element_size
        print("Add {0} prev {1} cur {2}".format(comp_name, total_size, added_size))
        total_size = total_size + added_size
    if comp_name in ['input_layernorm']:
        # 'post_attention_layernorm' could reuse the memory from input_layernorm
        added_size = seq_length * component_size[comp_name] * element_size
        print("Add {0} prev {1} cur {2}".format(comp_name, total_size, added_size))
        total_size = total_size + added_size
    if comp_name == 'mlp':
        # 3 hidden_size tensor should reuse qkv_proj
        added_size = (component_size[comp_name] * 2) * seq_length * element_size
        print("Add {0} prev {1} cur {2}".format(comp_name, total_size, added_size))
        total_size = total_size + added_size
        # total_size = (3 * hidden_size + component_size[comp_name] * 2) * seq_length * element_size + total_size
    if comp_name == 'rotary_emb':
        # assume that cos/sin table rope subgraph all runs in F32
        # matmul 
        total_size = total_size + seq_length * int(head_dim / 2) * 4
        # concat
        total_size = total_size + seq_length * head_dim * 4
        # cos_table
        total_size = total_size + seq_length * head_dim * 4
        # sin_table
        total_size = total_size + seq_length * head_dim * 4
        # rope k
        total_size = total_size + seq_length * component_size['k_proj'] * 2
        # rope q
        total_size = total_size + seq_length * component_size['q_proj'] * 2

input_size = seq_length * 4 * 2
kv_cache_size = seq_length * kv_hidden_size * num_hidden_layers
total_size = total_size + model_config['vocab_size'] * 4 + input_size + kv_cache_size
print("Total Runtime Memory {:.2f} GB".format(total_size / 1024 / 1024 / 1024))