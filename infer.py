import torch
import torch.nn as nn
from transformers import BigBirdModel, BigBirdTokenizerFast
from safetensors.torch import load_file
import numpy as np
import os

class DualChannelModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(DualChannelModel, self).__init__()
        self.bigbird = BigBirdModel.from_pretrained(model_name)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Learnable weight for text, starting fully weighted

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bigbird.config.hidden_size, self.bigbird.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.bigbird.config.hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, clue_input_ids, clue_attention_mask, labels=None):
        text_outputs = self.bigbird(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        clue_outputs = self.bigbird(input_ids=clue_input_ids, attention_mask=clue_attention_mask).last_hidden_state.mean(dim=1)

        # Calculate weighted fusion
        weights = torch.softmax(torch.stack([self.alpha, 1 - self.alpha]), dim=0)
        fused_features = text_outputs * weights[0] + clue_outputs * weights[1]

        logits = self.classifier(fused_features)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}

def load_model(checkpoint_path, model_name, num_labels=2):
    model = DualChannelModel(model_name, num_labels)
    model.bigbird.config.gradient_checkpointing = True
    state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device

def predict(model, device, tokenizer, texts, clues, batch_size=8):
    model.eval()
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_clues = clues[i:i+batch_size]

        text_inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=4096,
            return_tensors="pt"
        ).to(device)

        clue_inputs = tokenizer(
            batch_clues,
            padding="max_length",
            truncation=True,
            max_length=768,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                clue_input_ids=clue_inputs["input_ids"],
                clue_attention_mask=clue_inputs["attention_mask"]
            )
        probs = torch.softmax(outputs["logits"], dim=-1)[:, 1].cpu().numpy()
        predictions.extend(probs)
    return np.array(predictions)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference using the DualChannelModel.")
    parser.add_argument("--text", type=str, required=True, help="Input text for prediction.")
    parser.add_argument("--clues", type=str, required=True, help="Clues for the input text.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--model_name", type=str, default="google/bigbird-roberta-base", help="Name of the pre-trained model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")

    args = parser.parse_args()

    # Load model and tokenizer
    model, device = load_model(args.checkpoint_path, args.model_name)
    tokenizer = BigBirdTokenizerFast.from_pretrained(
        args.model_name,
        model_max_length=4096,
        padding_side="right",
        pad_to_multiple_of=64
    )

    # Prepare inputs
    texts = [args.text]
    clues = [args.clues]

    # Run prediction
    predictions = predict(model, device, tokenizer, texts, clues, batch_size=args.batch_size)

    # Output the result
    print(f"Prediction: {predictions[0]}")

if __name__ == "__main__":
    main()