from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import numpy as np

CONFIG ={
    'MODEL_PATH': "Qwen/Qwen2.5-14B", 
    'MISSION': 'hazard',
}

with open(CONFIG['MISSION'].upper(), 'r', encoding='utf-8') as f:
    CONFIG['MISSION_LABEL'] = eval(f.read())
    f.close()

def create_mission_tensor():
    model = AutoModelForCausalLM.from_pretrained(CONFIG['MODEL_PATH'], torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_PATH'], trust_remote_code=True)
    result = torch.tensor([])

    max_length = 0
    for i in tqdm(range(len(CONFIG['MISSION_LABEL']))):
        temp_label = CONFIG['MISSION_LABEL'][i] + ' class'
        temp_pt = tokenizer([temp_label], return_tensors='pt')
        with torch.no_grad():
            hidden_state = model(input_ids=temp_pt.input_ids.cuda(), attention_mask=temp_pt.attention_mask.cuda(), output_hidden_states=True).hidden_states[-1][:1, :, :]
            if max_length < int(hidden_state.size(1)):
                max_length = int(hidden_state.size(1))

            pad_tensor = torch.zeros([1, (1024-int(hidden_state.size(1))), hidden_state.size(2)]).cuda()
            hidden_state = torch.cat((hidden_state, pad_tensor), dim=1)
            result = torch.concat((result, hidden_state.detach().cpu().clone()), axis=0)
    result = result[:, :max_length, :]

    np.save('./'+CONFIG['MISSION']+'_tensor', np.array(result))
   
create_mission_tensor()