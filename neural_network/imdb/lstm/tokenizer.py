import sentencepiece as spm
import pandas as pd

df = pd.read_csv("./archive/IMDB Dataset SPANISH.csv")

reviews_es = df["review_en"].tolist()
spm.SentencePieceTrainer.Train(
    sentence_iterator=iter(reviews_es),
    model_prefix="imdb_model",
    vocab_size=25000,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)
sp = spm.SentencePieceProcessor(model_file="imdb_model.model")
