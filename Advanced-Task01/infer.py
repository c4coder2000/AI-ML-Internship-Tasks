from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

label_map ={
    0: "World",
    1: "Sci/Tech",
    2: "Sports",
    3: "Business"
}
model_path = "./bert-news-final"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    label_id = torch.argmax(logits, dim=1).item()
    return label_map.get(label_id, f"Unknown label {label_id}")
