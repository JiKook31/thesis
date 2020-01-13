from nltk.tag import pos_tag

with open("test.txt") as train_file:
    lines = train_file.readlines()

result = ""
for line in lines:
    splitted = line.split("\t")
    if len(splitted) > 1:
        word = splitted[0]
        ner_tag = splitted[1][:-1]
        pos = pos_tag([word])[0][1]
        result += f'{word} {pos} {ner_tag}\n'
    else:
        result += line

f = open("test_with_pos_table.txt", "w+")
f.write(result)
