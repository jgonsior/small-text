from transformers import AutoTokenizer
import datasets
import numpy as np
from small_text.integrations.transformers.datasets import TransformersDataset


def load_my_dataset(dataset: str, transformer_model_name: str):
    match dataset:
        case "20newsgroups":
            # nope
            raw_dataset = datasets.load_dataset("SetFit/20_newsgroups")
        case "ag_news":
            # works
            raw_dataset = datasets.load_dataset("ag_news")
        case "trec6":
            # works
            raw_dataset = datasets.load_dataset("trec")
            raw_dataset = raw_dataset.rename_column("label-coarse", "label")
        case "subj":
            # works
            raw_dataset = datasets.load_dataset("SetFit/subj")
        case "rotten":
            # works
            raw_dataset = datasets.load_dataset("rotten_tomatoes")
        case "imdb":
            # works
            raw_dataset = datasets.load_dataset("imdb")
        case _:
            print("dataset not known")
            exit(-1)

    print("First 3 training samples:\n")
    for i in range(3):
        print(raw_dataset["train"]["label"][i], " ", raw_dataset["train"]["text"][i])

    num_classes = np.unique(raw_dataset["train"]["label"]).shape[0]

    tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

    def _get_transformers_dataset(tokenizer, data, labels, max_length=60):

        data_out = []

        for i, doc in enumerate(data):
            encoded_dict = tokenizer.encode_plus(
                doc,
                add_special_tokens=True,
                padding="max_length",
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="pt",
                truncation="longest_first",
            )

            data_out.append(
                (encoded_dict["input_ids"], encoded_dict["attention_mask"], labels[i])
            )

        return TransformersDataset(data_out)

    # print(raw_dataset['train']['text'][:10])
    # print(raw_dataset['train']['label'][:10])

    train = _get_transformers_dataset(
        tokenizer, raw_dataset["train"]["text"], raw_dataset["train"]["label"]
    )
    test = _get_transformers_dataset(
        tokenizer, raw_dataset["test"]["text"], raw_dataset["test"]["label"]
    )

    return train, test, num_classes


def load_20newsgroups():
    # TODO: return validation for those who need validation!

    # Prepare some data: The data is a 2-class subset of 20news (baseball vs. hockey)
    text_train, text_test = get_train_test()
    train, test = preprocess_data(text_train, text_test)
    num_classes = 2
    return train, test, num_classes


def load_subj():
    pass


def load_rotten():
    pass


def load_imdb():
    pass
