import torch
import transformers
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import argparse
import deepspeed
from deepspeed.accelerator import get_accelerator

CONFIG ={
    'EPOCHS': 3,
    'LEARNING_RATE': 7e-6,
    'BATCH_SIZE_PER_GPU':4,
    'MAX_LENGTH': 8192,
    'WARMUP_RATE': 0.1, 
    'GPUS': 4,
    'GRADIENT_ACCUMULATION_STEPS': 4,
    'GRADIENT_CHECKPOINTING': True,
    'OUTPUT_DIR': "./output/",
    'MODEL_PATH': "Qwen/Qwen2.5-14B",
    
    'SYSTEM': 'You are a helpful assistant.',
    'TRAIN_DATA_PATH': ["../data/data/product_llama_factory.json"],
    'EMBEDDING_PATH': '../data/embeddings/product_tensor.npy',
}

def seed_everything(seed=42):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(CONFIG['MODEL_PATH'], torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.hidden_size = self.model.config.hidden_size

        if CONFIG['GRADIENT_CHECKPOINTING'] == True:
            self.model.gradient_checkpointing_enable()
        
        self.mission_tensor = torch.tensor(np.load(CONFIG['EMBEDDING_PATH']), dtype=torch.bfloat16).cuda()

        self.WQ = nn.Linear(self.hidden_size, self.hidden_size * 12, bias=False)
        self.WKV = nn.Linear(self.hidden_size, self.hidden_size * 12, bias=False)
        self.WO = nn.Linear(self.hidden_size*12, self.hidden_size)
        self.FFN = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm = nn.LayerNorm([self.model.config.hidden_size], dtype=torch.bfloat16)

    
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        bz = output.size(0)
        length = output.size(1)
        output = output.reshape(bz*length, -1)

        Q = self.WQ(output) # bz*length, hidden*12
        KV = self.WKV(self.mission_tensor.cuda()) # class, text, hidden*12
        class_number = KV.size(0)
        text_number = KV.size(1)
        KV = KV.reshape(class_number*text_number, -1) # class*text, hidden*12

        QK = torch.matmul(Q, KV.T) # bz*length, class*text
        QK_d = QK / (self.hidden_size**0.5) # bz*length, class*text
        QK_d = QK_d.reshape(bz*length, class_number, text_number) # bz*length, class, text

        sQK_d = torch.softmax(QK_d, dim=2) # bz*length, class, text
        sQK_d = sQK_d.reshape(bz*length, class_number*text_number) # bz*length, class*text
 
        QKV = torch.matmul(sQK_d, KV) # bz*length, hidden*12
        QKV = self.WO(QKV) # bz*length, hidden
      
        output = QKV + output
        output = self.LayerNorm(output) # bz*length, hidden

        output = self.FFN(output) + output
        output = self.LayerNorm(output) # bz*length, hidden
        
        logits = self.model.lm_head(output) # bz*length, vocab
        logits = logits.reshape(bz, length, -1) # bz, length, vocab
     
        return logits

class MyDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.datacollatorforseq2seq = transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)
        self.im_start = 151644
        self.im_end = 151645
        self.system = 8948
        self.user = 872
        self.assistant = 77091
        self._n = 198
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.df)
    
    def tokenize(self, x, role):
        ids = self.tokenizer(x)['input_ids']
        if role == 'system':
            ids = [self.im_start, self.system, self._n] + ids + [self.im_end, self._n]
        elif role == 'user':
            ids = [self.im_start, self.user, self._n] + ids + [self.im_end, self._n, self.im_start, self.assistant, self._n]
        elif role == 'assistant':
            ids = ids + [self.im_end]
        return ids

    def preprocess(self, data):
        encoded_data = []
        label = []
        encoded_data += self.tokenize(CONFIG['SYSTEM'], 'system')
        label += [-100] * len(encoded_data)
        for ind, d in enumerate(data):
            if ind %2 == 0:
                temp_ids = self.tokenize(d, 'user')
                encoded_data += temp_ids.copy()
                label += [-100] * len(temp_ids)
            else:
                temp_ids = self.tokenize(d, 'assistant')
                encoded_data += temp_ids.copy()
                label += temp_ids.copy()
       
        return {'input_ids': encoded_data[:CONFIG['MAX_LENGTH']], 'labels': label[:CONFIG['MAX_LENGTH']]}
    def __getitem__(self, index):
        return self.df.iloc[index]['data']


    def collate_fn(self, batch):
        processed_batch = [self.preprocess(x) for x in batch]
        return self.datacollatorforseq2seq(processed_batch)

def set_optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    return optimizer

def set_data():
    train_data = pd.DataFrame(columns=['query', 'response'])
    for temp_data_path in CONFIG['TRAIN_DATA_PATH']:
        with open(temp_data_path, 'r', encoding='utf-8') as f:
            temp_data = eval(f.read())
            f.close()
        for temp_dict in temp_data:
            temp_series = pd.Series({'query': temp_dict['instruction'], 'response': temp_dict['output']})
            train_data.loc[len(train_data)] = temp_series
   
    train_data = train_data.sample(frac=1.0).reset_index(drop=True)

    result_train_data = pd.DataFrame(columns=['data'])
    for i in tqdm(range(len(train_data))):
        temp = [train_data.iloc[i]['query'], train_data.iloc[i]['response']]
        temp_series = pd.Series({'data': temp.copy()})
        result_train_data.loc[len(result_train_data)] = temp_series

    return result_train_data

def set_lr_scheduler(len_train_dataset, optimizer):
    num_training_steps = len_train_dataset * CONFIG['EPOCHS'] // CONFIG['GRADIENT_ACCUMULATION_STEPS'] // CONFIG['BATCH_SIZE_PER_GPU'] // CONFIG['GPUS']

    lr_scheduler = deepspeed.runtime.lr_schedules.WarmupCosineLR(optimizer=optimizer, total_num_steps=num_training_steps, warmup_num_steps=int(CONFIG['WARMUP_RATE'] * num_training_steps))
  
    return lr_scheduler
    
def train():
    seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_PATH'], trust_remote_code=True)
    
    model = MyModel()
   
    optimizer = set_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    train_data = set_data()
    train_dataset = MyDataset(train_data, tokenizer)

    lr_scheduler = set_lr_scheduler(len(train_dataset), optimizer)
    model, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(args=cmd_args, model=model, optimizer=optimizer, training_data=train_dataset, collate_fn=train_dataset.collate_fn, lr_scheduler=lr_scheduler)
    
    torch.cuda.empty_cache()
    for epoch in range(CONFIG['EPOCHS']):
        model.train() 
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if cmd_args.local_rank==0 else enumerate(train_dataloader)
        for i, data in train_dataloader_iterator:
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            labels = data['labels']
           
            logits =model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())
            
            logits = logits[:, :-1, :]            
            bz = logits.size(0)
            length = logits.size(1)
            logits = logits.reshape(bz*length, -1)
            labels = labels[:, 1:].reshape(-1)
         
            loss = criterion(logits, labels.cuda())
            print('<loss>:', loss, flush=True)
  
            if cmd_args.local_rank==0:
                print('<lr>:', optimizer.param_groups[0]['lr'])
     
            model.backward(loss)
    
            model.step()
            model.empty_partition_cache()
            torch.cuda.empty_cache()
            get_accelerator().empty_cache()

        model.save_16bit_model(CONFIG['OUTPUT_DIR']+str(epoch))
        model.empty_partition_cache()
        torch.cuda.empty_cache()
        get_accelerator().empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()
    
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    train()