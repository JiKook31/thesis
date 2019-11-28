from nltk.tag import pos_tag

with open("test.txt") as train_file:
    lines = train_file.readlines()

result = []
current_sentence = []
for line in lines:
    splitted = line.split("\t")
    if len(splitted) > 1:
        word = splitted[0]
        ner_tag = splitted[1][:-1]
        pos = pos_tag([word])[0][1]
        current_sentence.append((word, pos, ner_tag))
    else:
        result.append(current_sentence)
        current_sentence = []

f = open("test_with_pos.txt", "w+")
f.write(str(result))

