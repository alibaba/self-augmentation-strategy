# coding=utf-8
import os
import pickle
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BertTokenizer
import numpy as np 


def generate_dataset(tokenizer: PreTrainedTokenizer, file_path: str, sequence_length: int):
    """
    Input : BertTokenizer, and wiki_books_corpus_data 
    Output : compressed wiki corpus data stored in numpy.int16.
    """
    directory, filename = os.path.split(file_path)
    cached_features_file = os.path.join(
        directory, "mlm_{}_{}_{}".format(tokenizer.__class__.__name__, str(sequence_length), filename))
    assert os.path.isfile(file_path)
    examples = []
    current_block = []
    current_length = 0
    idx = 0
    with open(file_path, encoding="utf-8") as f:
        while True:
            idx += 1
            print(idx)
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                current_block = []
                current_length = 0
                continue
            sentence = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
            current_block.extend(sentence)
            current_length += len(sentence)
            if current_length >= sequence_length:
                examples.append(current_block)
                current_block = []
                current_length = 0

        print(examples[:5])
        with open(cached_features_file, "wb") as handle:
            pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)


def slice_to_length(path="mlm_BertTokenizer_1024_wiki_books_corpus_data_train", length=128):
    """
    Input : Tokenized wiki_books_corpus data 
    Output : compressed wiki corpus data stored in numpy.int16.
    """

    with open(path, 'rb') as f:
        data = pickle.load(f)
    output = []
    for i in range(128 // length):
        data128 = np.array([col[length*i:length*i+length] for col in data 
                    if len(col) >= length*i+length], dtype=np.int16)
        output.append(data128)
    output = np.concatenate(output)
    print(output.shape)
    np.save("wiki_corpus_full", output)

def calculat_unigram():
    """
    Count total number of tokens. Only support 30K word-version dataset. 
    Input : A npy file with arbitrary size.
    Output: An array include number of tokens.
    """

    import numpy as np
    examples = np.load("dataset/wiki_corpus_full.npy")
    np.random.shuffle(examples)
    unique_elements, empirical_distribution = np.unique(examples, return_counts=True)
    dis = np.zeros((30522))
    dis[unique_elements] = empirical_distribution
    sst = np.argsort(dis)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(["%s: %d" % (tokenizer.decode([s]), dis[s]) for s in sst[-1000:]])
    np.save("unigram.npy", dis)


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    generate_dataset(tokenizer, 'dataset/test', 1024)
    slice_to_length('dataset/mlm_BertTokenizer_1024_test')




