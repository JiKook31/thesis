from deeppavlov import configs, build_model

class color:
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

ner_model = build_model(configs.ner.ner_conll2003_bert, download=False)

with open("input.txt") as file:
    text = file.read()
answer = ner_model([text])
splitted_text = answer[0][0]
entities = answer[1][0]
entity_text = ''
for i, entity in enumerate(entities):
    if entity != 'O':
        entity_text += f'{color.BOLD}{splitted_text[i]} ({entity}{color.END}) '
    else:
        entity_text += f'{splitted_text[i]} '
print(entity_text)
