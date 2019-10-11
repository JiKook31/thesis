import os
from pathlib import Path
from deeppavlov import configs, build_model, train_model

deeppavlov_path = f"{Path('~/.deeppavlov').expanduser()}/downloads/conll2003"
model_path = f"{Path('~/.deeppavlov').expanduser()}/models/ner_conll2003_bert"

if not os.path.isdir(model_path):
    ner_model = build_model(configs.ner.ner_conll2003_bert, download=True)
    print("DOWNLOADING IS FINISHED")

remove_initial_datasets = f'mkdir {deeppavlov_path}'
insert_custom_datasets = f'cp ./test.txt {deeppavlov_path}/test.txt\n' \
                         f'cp ./train.txt {deeppavlov_path}/train.txt'
remove_model_data = f'rm {model_path}/model.*'

os.system(remove_initial_datasets)
os.system(insert_custom_datasets)
os.system(remove_model_data)
print("ALL FILES ARE MOVED")

print("STARTING TRAINING")
train_model(configs.ner.ner_conll2003_bert, download=False)
