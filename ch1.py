import re
from simple_tokenizer import SimpleTokenizer
from simple_tokenizerv2 import SimpleTokenizerV2

with open("the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print ("Total chars:", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}
# (0: "hello"), (1: "world"), (2: "...")
# ("hello": 0), ("world", 1), ("...", 2)
for i, item in enumerate(vocab.items()):
    # (0, ("hello": 0)), (1, ("world", 1)), ("...", 2)
    # ('!', 0)
    # ('a', 1)
    # ('b', 2)

    print(item)
    if i > 50:
        break

####

# masterful code

for item in vocab.items(): 
    while item[1] <= 51:
        print(item)
        break

#####

tokenizer = SimpleTokenizer(vocab)
text = """"It's the last he painted, you know,"
 Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# ["I", "love", "<unk>", "and", "<unk>"]\
# [10, 57, 1131, 42, 1131]

vocab = {token:integer for integer,token in enumerate(all_tokens)}
# (0, 'a'), (1, 'b'), ...
# ('a', 0), ('b', 1), ...

for item in vocab.items():
    print(item)
    if item[1] == 10:
        break

for i, item in enumerate(list(vocab.items())[-5:]):
 print(item)


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
# [1131, 5, 355, 1126, 628, 975, 10, 1131, 55, 988, 956, 984, 722, 988, 1131, 7]

print(tokenizer.decode(tokenizer.encode(text)))
# <|unk|>, do you like tea? <|unk|> In the sunlit terraces of the <|unk|>.