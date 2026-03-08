import sentencepiece as spm
import pandas as pd

df = pd.read_csv("./archive/IMDB Dataset SPANISH.csv")

# Tomamos solo la columna en español
reviews_es = df["review_es"].tolist()
spm.SentencePieceTrainer.Train(
    sentence_iterator=iter(reviews_es),
    model_prefix="imdb_model",
    vocab_size=15000,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)
sp = spm.SentencePieceProcessor(model_file="imdb_model.model")
sentence = "Esta fue una película muy mala, el guion era débil y la actuación peor."
prueba = sp.EncodeAsIds(sentence)
print(sp.DecodeIds(prueba))
