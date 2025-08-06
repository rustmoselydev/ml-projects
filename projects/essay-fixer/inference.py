from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

mock_essay = "I was going to store yesterday when a friend me saw. He said, 'You look very nice today!' I wondered how he could thought me looked this nice when me was only simple caveman. I found his compliments to be quite flattering. Me wandered on around store and think me maybe handsome. It's difficult being a caveman in modern society, but at least I have good friends."
sentences = sent_tokenize(mock_essay)
# 3 sentence groupings
phrases = ["".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]


acceptability = pipeline("text-classification", model="textattack/roberta-base-CoLA")
corrector = pipeline(
              'text2text-generation',
            #   'hassaanik/grammar-correction-model'
              'pszemraj/flan-t5-large-grammar-synthesis',
              )
final_text = ""

for phrase in sentences:
    score = acceptability(phrase)
    if score[0]["label"] == "LABEL_0":
        results = corrector(phrase)
        final_text += results[0]["generated_text"] + " "
    else:
        final_text += phrase + " "

print(final_text)



