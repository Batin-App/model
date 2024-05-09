import os
from collections import OrderedDict

import boto3
import torch
from flask import Flask, request, jsonify
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

app = Flask(__name__)


class EmotionClassifierWithConv(nn.Module):
    def __init__(self, transformer_model, num_classes, kernel_size=3, num_filters=256):
        super(EmotionClassifierWithConv, self).__init__()
        self.transformer = transformer_model
        self.conv = nn.Conv1d(in_channels=768, out_channels=num_filters, kernel_size=kernel_size,
                              padding=1)  # Adjust padding
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        pooled_output = pooled_output.unsqueeze(2)

        conv_out = F.relu(self.conv(pooled_output))
        pooled_conv_out, _ = torch.max(conv_out, dim=2)
        logits = self.fc(pooled_conv_out)
        return logits


device = torch.device("cpu")


def load_model():
    model = AutoModel.from_pretrained("bert-base-uncased")

    model = EmotionClassifierWithConv(model, 6)

    model_state_dict = torch.hub.load_state_dict_from_url("https://assets.djaeger.dev/fine_tuned_batin.pth", map_location=device)

    if next(iter(model_state_dict.keys())).startswith("module."):
        new_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            name = key[7:]  # remove "module." prefix
            new_state_dict[name] = value
        model_state_dict = new_state_dict

    # Load the state_dict into the model
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


prediction_model, prediction_tokenizer = load_model()


@app.route('/', methods=['POST'])
def predict_emotion():  # put application's code here
    data = request.get_json()

    text = data['text']

    inputs = prediction_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = prediction_model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities.tolist()})


if __name__ == '__main__':
    app.run()
