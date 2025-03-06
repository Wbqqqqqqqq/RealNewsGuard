import transformers
import warnings

# Suppress warnings related to tokenizers and model initialization
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import torch
import torch.nn as nn
import os
import numpy as np
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer, RagRetriever, RagSequenceForGeneration, DPRQuestionEncoder, BigBirdModel
)

def initialize_rag_components(model_name):
    """
    Initialize RAG components:
      - Load the Tokenizer (AutoTokenizer to prevent mismatch)
      - Load the DPR Question Encoder (used for query encoding)
      - Initialize the Retriever (FAISS vector database and news dataset)
      - Load the RAG model and inject the Retriever
    """
    # 1. Load Tokenizer with AutoTokenizer (Fixes mismatch warnings)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 2. Load DPR Question Encoder and its corresponding Tokenizer
    question_encoder_model = "facebook/dpr-question_encoder-single-nq-base"
    question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_model)
    question_tokenizer = AutoTokenizer.from_pretrained(question_encoder_model, trust_remote_code=True)

    # 3. Initialize Retriever (based on FAISS vector database)
    base_path = os.path.abspath("assets/Retriever")
    index_path = os.path.join(base_path, "custom_index.faiss")
    passages_path = os.path.join(base_path, "news_dataset_with_emb")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not os.path.exists(passages_path):
        raise FileNotFoundError(f"News dataset not found at {passages_path}")

    retriever = RagRetriever.from_pretrained(
        model_name,
        index_name="custom",
        index_path=index_path,
        passages_path=passages_path,
    )
    # print("✅ Retriever initialized!")

    # 4. Load RAG model and inject the Retriever
    model = RagSequenceForGeneration.from_pretrained(
        model_name,
        retriever=retriever
    )

    return tokenizer, question_encoder, question_tokenizer, retriever, model


def retrieve_clues(text, tokenizer, model, retriever, n_docs=5):
    """
    Perform CLUE retrieval for a single text:
      - Compute query embeddings (RAG question encoder)
      - Retrieve relevant documents from FAISS
      - Return the most relevant CLUES (retrieved documents)
    """
    query = f"Retrieve news articles related to: {text}"

    # 1️⃣ Compute query embeddings
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    question_encoder_outputs = model.question_encoder(input_ids)
    question_hidden_states = question_encoder_outputs[0]
    question_hidden_states_np = question_hidden_states.detach().cpu().numpy()

    # 2️⃣ Retrieve relevant documents
    retrieved_doc_dict = retriever(
        question_input_ids=input_ids,
        question_hidden_states=question_hidden_states_np,
        n_docs=n_docs,
    )
    doc_ids = retrieved_doc_dict["doc_ids"]
    doc_ids_np = np.array(doc_ids[0].tolist())
    doc_dicts = retriever.index.get_doc_dicts(doc_ids_np)
    retrieved_docs = [doc["text"] for doc in doc_dicts]

    # Format the clues as "Clue 1: ..." "Clue 2: ..."
    formatted_clues = "\n".join([f"Clue {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])

    return formatted_clues

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
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--model_name", type=str, default="google/bigbird-roberta-base", help="Name of the pre-trained model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")

    args = parser.parse_args()

    # Load model and tokenizer
    model, device = load_model(args.checkpoint_path, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=4096,
        padding_side="right",
        pad_to_multiple_of=64,
        trust_remote_code=True  # Fixes tokenizer mismatch warnings
    )

    # Load RAG components for retrieving clues
    rag_model_name = "facebook/rag-sequence-base"
    rag_tokenizer, _, _, retriever, rag_model = initialize_rag_components(rag_model_name)

    # Prepare inputs
    texts = [args.text]
    
    # Retrieve relevant clues for the input text
    clues = [retrieve_clues(args.text, rag_tokenizer, rag_model, retriever, n_docs=10)]

    # Run prediction
    predictions = predict(model, device, tokenizer, texts, clues, batch_size=args.batch_size)

    # Output the result
    print(f"\nPrediction: {predictions[0]}")

if __name__ == "__main__":
    main()