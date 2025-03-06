import os
import json
import nltk
import faiss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
)

# Confirm faiss version
print("faiss imported successfully, version:", faiss.__version__)


def load_csv_dataset(dataset_path):
    """Load CSV dataset using pandas."""
    return pd.read_csv(dataset_path)


def chunk_and_tokenize_documents(dataset):
    """
    Split each document from the DataFrame into chunks using a text splitter,
    tokenize the chunks, and accumulate token counts.
    """
    news_docs = []
    total_tokens = 0

    # Initialize text splitter parameters
    chunk_size = 1536  # Each chunk's character count (not token count)
    chunk_overlap = 512  # Allowed overlapping characters to preserve semantics
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ".", "!", "?"],
    )

    # Iterate over DataFrame rows
    for index, row in dataset.iterrows():
        title = str(row["title"])
        text = str(row["content"])

        # Split the text into chunks
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            news_docs.append({"title": title, "text": chunk})
            total_tokens += len(nltk.word_tokenize(chunk))

    return news_docs, total_tokens


def save_json(data, output_path):
    """Save the data to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def encode_texts(texts, tokenizer, model, device, batch_size=64):
    """
    Encode texts in batches using the provided tokenizer and model.
    Returns the concatenated embeddings.
    """
    embeddings_list = []
    pbar = tqdm(total=len(texts), desc="Encoding batches", leave=True)

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            try:
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
            except Exception as e:
                print("Exception:", e)
                print("Error processing batch_texts:", batch_texts)
            # Move inputs to the device
            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)
            # outputs.pooler_output shape: (batch_size, hidden_dim)
            embeddings = outputs.pooler_output.cpu().numpy()  # Move to CPU after computation
            embeddings_list.append(embeddings)
            pbar.update(len(batch_texts))
    pbar.close()

    # Concatenate embeddings from all batches
    embeddings_all = np.concatenate(embeddings_list, axis=0)
    return embeddings_all


def build_and_save_faiss_index(embeddings_all, index_path="custom_index.faiss"):
    """
    Normalize the embeddings, build a FAISS index using inner product,
    and save the index to disk.
    """
    faiss.normalize_L2(embeddings_all)
    d = embeddings_all.shape[1]  # Vector dimensionality (usually 768)
    index = faiss.IndexFlatIP(d)  # Use inner product similarity
    index.add(embeddings_all)
    print(f"FAISS index now contains {index.ntotal} records")
    faiss.write_index(index, index_path)
    print("Custom FAISS index has been saved as", index_path)
    return index


def add_embeddings_to_dataset(dataset_path, embeddings_all):
    """
    Load a dataset from a JSON file, add an embeddings column,
    and then save the updated dataset to disk.
    """
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print("Current dataset columns:", dataset.column_names)
    dataset = dataset.add_column("embeddings", embeddings_all.tolist())
    dataset.save_to_disk("news_dataset_with_emb")
    print("Dataset with embeddings saved to disk as 'news_dataset_with_emb'.")


def main():
    # ----------------------------
    # Step 1: Load and process CSV dataset
    # ----------------------------
    csv_path = r"D:\Desktop\SI630\Project\RealNewsGuard\data\Retriever Dataset\nytimes_cleaned_data_2020.csv"
    dataset = load_csv_dataset(csv_path)

    news_docs, total_tokens = chunk_and_tokenize_documents(dataset)
    num_documents = len(news_docs)
    avg_tokens_per_chunk = total_tokens / num_documents if num_documents > 0 else 0

    print(f"Total number of documents: {num_documents}")
    print(f"Average tokens per chunk: {avg_tokens_per_chunk:.2f}")

    # Save processed news documents to JSON
    json_output_path = "news_dataset.json"
    save_json(news_docs, json_output_path)
    print("Dataset saved successfully. File path:", json_output_path)
    if num_documents > 1:
        print("Sample Docs: ", news_docs[1])

    # ----------------------------
    # Step 2: Train news dataset embeddings
    # ----------------------------
    # Load custom news dataset from the JSON file
    with open(json_output_path, "r", encoding="utf-8") as f:
        news_data = json.load(f)
    texts = [doc["text"] for doc in news_data]

    # Load DPR context encoder and tokenizer
    ctx_encoder_model = "facebook/dpr-ctx_encoder-single-nq-base"
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_encoder_model)
    model = DPRContextEncoder.from_pretrained(ctx_encoder_model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Batch encode news texts using GPU acceleration
    embeddings_all = encode_texts(texts, tokenizer, model, device, batch_size=64)
    print(f"Shape of generated embeddings: {embeddings_all.shape}")

    # Build and save FAISS index
    build_and_save_faiss_index(embeddings_all, index_path="custom_index.faiss")

    # Add embeddings column to dataset and save
    add_embeddings_to_dataset(json_output_path, embeddings_all)


if __name__ == "__main__":
    main()
