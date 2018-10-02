import json
import pathlib
import collections

chitchat_file = pathlib.Path("/mnt/pccfs/not_backed_up/data/chitchat/processed/ccc_data_flat.json")
chitchat_data = json.loads(chitchat_file.read_text())

ENDING_TAG = "#END_OF_CONVERSATION"

all_sentences = [wrapped_sentence[0] for wrapped_sentence in chitchat_data if wrapped_sentence[0] != ENDING_TAG]

# collect action space
counter = collections.Counter(" ".join(all_sentences))

# print(counter)

# count_pairs = sorted(counter.items(), key=lambda x: -x[1])
# chars, _ = zip(*count_pairs)
# chars = list(chars)
# vocab_size = len(chars)
# vocab = dict(zip(chars, range(len(chars))))


a = "😍"
b = "d"
print(type(a))
print(type(b))