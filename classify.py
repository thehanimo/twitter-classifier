from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import expit


def classify(text):
  MODEL = f"cardiffnlp/tweet-topic-21-multi"
  tokenizer = AutoTokenizer.from_pretrained(MODEL)

  # PT
  model = AutoModelForSequenceClassification.from_pretrained(MODEL)
  class_mapping = model.config.id2label
  tokens = tokenizer(text, return_tensors='pt')
  output = model(**tokens)

  scores = output[0][0].detach().numpy()
  scores = expit(scores)
  predictions = (scores >= 0.5) * 1


  # TF
  #tf_model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
  #class_mapping = tf_model.config.id2label
  #text = "It is great to see athletes promoting awareness for climate change."
  #tokens = tokenizer(text, return_tensors='tf')
  #output = tf_model(**tokens)
  #scores = output[0][0]
  #scores = expit(scores)
  #predictions = (scores >= 0.5) * 1

  out = []
  for i in range(len(predictions)):
    if predictions[i]:
      out.append(class_mapping[i])
  
  return out