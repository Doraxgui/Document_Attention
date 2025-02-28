import pandas as pd
from transformers import AutoTokenizer
import json
from tqdm import tqdm
CONFIG = {
    'MODEL_PATH': 'Qwen/Qwen2.5-14B',
    'SYSTEM': 'You are a helpful assistant.',
    'DATA_PATH': 'incidents_train.csv'
}

tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_PATH'])

def data_clean(text):
    text = text.lower()
    result = []
    for temp in text.split():
        if temp not in result:
            result.append(temp)
    result = ' '.join(result)
    return result

for mission in ['hazard', 'hazard-category', 'product', 'product-category']:
    dev_data = pd.read_csv(CONFIG['DATA_PATH'])

    result = []
    query = '\n</Text>\n</Context>\n\n<Question>\nWhich ' + mission + ' does the food incidents of <Context> belong to?\n</Question>\n\nAccording to <Context>, answer the <Question>. Only Answer the option, do not explain the reason.' 

    for i in tqdm(range(len(dev_data))):
        output = dev_data.iloc[i][mission]
        prompt = '<Context>\n<Title>\n' + dev_data.iloc[i]['title'] + '\n</Title>\n<Text>\n' + data_clean(dev_data.iloc[i]['text'])

        temp_tokens = tokenizer(prompt)['input_ids']      
        if len(temp_tokens) > 1024:
            prompt = tokenizer.decode(temp_tokens[:1024])
        prompt += query

        temp_result = {'input': ""}
        temp_result['instruction'] = prompt
        temp_result['output'] = output
        temp_result['system'] = CONFIG['SYSTEM']
        temp_result['history'] = []
        result.append(temp_result.copy())

    with open('./'+mission+'_llama_factory.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.close()        