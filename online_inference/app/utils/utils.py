import pickle


def load_pkl_file(input_: str):
    with open(input_, "rb") as fin:
        res = pickle.load(fin)
    return res
