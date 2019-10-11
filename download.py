import os
from pathlib import Path
from deeppavlov import configs, build_model

deeppavlov_path = f"{Path('~/.deeppavlov').expanduser()}/downloads/conll2003"
if not os.path.isdir(deeppavlov_path):
    ner_model = build_model(configs.ner.ner_conll2003_bert, download=True)
    print("DOWNLOADING IS FINISHED")