import os
from pathlib import Path
from deeppavlov import configs, train_model

deeppavlov_path = f"{Path('~/.deeppavlov').expanduser()}/downloads/conll2003"
model_path = f"{Path('~/.deeppavlov').expanduser()}/models/ner_conll2003_bert"

remove_initial_datasets = f'rm {deeppavlov_path}/*.txt'
insert_custom_datasets = f'cp ./test.txt {deeppavlov_path}/test.txt\n' \
                         f'cp ./train.txt {deeppavlov_path}/train.txt'
remove_model_data = f'rm {model_path}/model.*'

os.system(remove_initial_datasets)
os.system(insert_custom_datasets)
os.system(remove_model_data)
print("ALL FILES ARE MOVED")
os.system(f'ls {deeppavlov_path}')

print("STARTING TRAINING")
train_model(configs.ner.ner_conll2003_bert, download=False)