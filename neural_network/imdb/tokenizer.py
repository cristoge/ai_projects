import sentencepiece as spm

x = [
    "Esta película fue fantástica, la actuación fue excelente y la historia muy atractiva.",
    "Realmente me encantó esta película, los personajes eran interesantes y la trama emocionante.",
    "Una película increíble con actuaciones brillantes y una banda sonora hermosa.",
    "Esta película fue aburrida y demasiado larga, casi me quedo dormido.",
    "Una película terrible con mala actuación y una historia predecible.",
    "Disfruté la película, fue entretenida y bien dirigida.",
    "La película empezó bien pero el final fue decepcionante.",
    "Una película absolutamente maravillosa, definitivamente la volvería a ver.",
    "Esta fue una película muy mala, el guion era débil y la actuación peor.",
    "Una película decente con algunos buenos momentos pero en general promedio.",
]
spm.SentencePieceTrainer.Train(
    sentence_iterator=iter(x),
    model_prefix="imdb_model",
    vocab_size=100,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)
sp = spm.SentencePieceProcessor(model_file="imdb_model.model")
sentence = "Esta fue una película muy mala, el guion era débil y la actuación peor."
prueba = sp.EncodeAsIds(sentence)
print(sp.DecodeIds(prueba))
