import re
from tokenizer import SimpleTokenizer

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
all_tokens.extend([])