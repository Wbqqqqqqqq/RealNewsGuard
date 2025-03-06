import os
import json
import torch
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
)
from rank_bm25 import BM25Okapi

print("faiss imported successfully, version:", faiss.__version__)


def initialize_rag_components(model_name):
    """
    Initialize RAG components including:
      - Loading the Tokenizer (RAG requires RagTokenizer, not DPRContextEncoderTokenizer)
      - Loading the DPR Question Encoder (RAG uses DPRQuestionEncoder for query encoding instead of DPRContextEncoder)
      - Initializing the Retriever with a custom index, specifying the FAISS vector database path and pre-stored news dataset
      - Loading the RAG model and injecting the retriever
    """
    # 1. Load Tokenizer
    tokenizer = RagTokenizer.from_pretrained(model_name)

    # 2. Load DPR Question Encoder and its tokenizer
    question_encoder_model = "facebook/dpr-question_encoder-single-nq-base"
    question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_model)
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_encoder_model)
    question_encoder.eval()

    # 3. Initialize Retriever
    retriever = RagRetriever.from_pretrained(
        model_name,
        index_name="custom",  # Custom index
        index_path="custom_index.faiss",  # FAISS vector database path
        passages_path="news_dataset_with_emb",  # Pre-stored news dataset
    )
    print("Retriever initialized!")

    # 4. Load RAG model and inject the retriever
    model = RagSequenceForGeneration.from_pretrained(
        model_name,
        retriever=retriever
    )

    return tokenizer, question_encoder, question_tokenizer, retriever, model


def process_dataset(df, dataset_name, tokenizer, model, retriever):
    """
    For a given DataFrame:
      - Iterate over each sample to compute query embeddings using the RAG model
      - Retrieve relevant documents
      - Compute BM25 similarity
      - Store the retrieved documents and BM25 scores
      - Add the 'clue' and 'bm25' columns (keeping 'text' and 'generated') and save as a Parquet file
    """
    retrieved_docs_list = []
    bm25_scores_list = []

    for idx in tqdm(range(len(df)), desc=f"Processing samples for {dataset_name}"):
        sample = df.iloc[idx:idx+1]
        query = "Retrieve news articles related to: " + sample["text"].values[0]

        # 1️⃣ Compute query embeddings using the RAG model
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        question_encoder_outputs = model.question_encoder(input_ids)
        question_hidden_states = question_encoder_outputs[0]
        question_hidden_states_np = question_hidden_states.detach().cpu().numpy()

        # 2️⃣ Retrieve relevant documents
        retrieved_doc_dict = retriever(
            question_input_ids=input_ids,
            question_hidden_states=question_hidden_states_np,
            n_docs=10,
        )
        doc_ids = retrieved_doc_dict["doc_ids"]
        doc_ids_np = np.array(doc_ids[0].tolist())
        doc_dicts = retriever.index.get_doc_dicts(doc_ids_np)
        retrieved_docs = [doc["text"] for doc in doc_dicts]

        # Compute BM25 similarity
        tokenized_corpus = [doc.split() for doc in retrieved_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Store data
        retrieved_docs_list.append(retrieved_docs)
        bm25_scores_list.append(bm25_scores.tolist())

    # Add 'clue' and 'bm25' columns, keeping only 'index', 'text', and 'generated'
    df["clue"] = retrieved_docs_list
    df["bm25"] = bm25_scores_list
    df = df[["index", "text", "generated", "clue", "bm25"]]

    # Save the DataFrame as a Parquet file
    parquet_file_path = f"RAG_results_{dataset_name}.parquet"
    df.to_parquet(parquet_file_path, index=False)
    print(f"Saved processed dataset to {parquet_file_path}")


def compute_features(scores):
    """
    Compute features for BM25 scores.
    Returns a pandas Series with the following features:
      - mean: average score
      - top3_mean: mean of the top 3 scores
      - top5_mean: mean of the top 5 scores
      - best: highest score
      - median: median score
      - covariance: variance of scores with Bessel's correction (ddof=1)
      - max_ratio: best score divided by the sum of scores
      - iqr: interquartile range (Q3 - Q1)
    """
    if scores is None or (isinstance(scores, (list, np.ndarray, pd.Series)) and len(scores) < 5):
        return pd.Series({
            'mean': np.nan,
            'top3_mean': np.nan,
            'best': np.nan,
            'median': np.nan,
            'covariance': np.nan
        })

    scores_sorted = sorted(scores, reverse=True)
    mean_value = np.mean(scores)
    top3_mean = np.mean(scores_sorted[:3]) if len(scores) >= 3 else mean_value
    top5_mean = np.mean(scores_sorted[:5]) if len(scores) >= 5 else mean_value
    best_value = scores_sorted[0]
    median_value = np.median(scores)
    covariance_value = np.var(scores, ddof=1) if len(scores) > 1 else np.nan
    max_ratio = best_value / sum(scores) if sum(scores) > 0 else np.nan
    q1, q3 = np.percentile(scores, [25, 75])
    iqr_value = q3 - q1

    return pd.Series({
        'mean': mean_value,
        'top3_mean': top3_mean,
        'top5_mean': top5_mean,
        'best': best_value,
        'median': median_value,
        'covariance': covariance_value,
        'max_ratio': max_ratio,
        'iqr': iqr_value,
    })


def extract_features_from_files(file_paths):
    """
    For each file in the list:
      - Check if the file exists
      - If it exists, load the Parquet file and ensure the 'bm25' column is present
      - Compute BM25 features using the compute_features function
      - Save the resulting features as a new Parquet file with '_features' appended to the filename
    """
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
        else:
            print(f"Processing file: {file_path}")
            df = pd.read_parquet(file_path)

            if 'bm25' not in df.columns:
                print(f"Column 'bm25' not found in {file_path}")
                continue

            df_features = df['bm25'].apply(compute_features)
            output_file = file_path.replace(".parquet", "_features.parquet")
            df_features.to_parquet(output_file)
            print(f"Feature extraction completed. Saved to {output_file}")


def main():
    # Initialize RAG components
    model_name = "facebook/rag-sequence-base"
    tokenizer, question_encoder, question_tokenizer, retriever, model = initialize_rag_components(model_name)

    # Load datasets
    test1 = pd.read_csv("D:\Desktop\SI630\Project\RealNewsGuard\data\Dataset\test1_final.csv")
    test2 = pd.read_csv("D:\Desktop\SI630\Project\RealNewsGuard\data\Dataset\test2_final.csv")
    train = pd.read_csv("D:\Desktop\SI630\Project\RealNewsGuard\data\Dataset\train_final.csv")

    datasets = {
        "train": train,
        "test1": test1,
        "test2": test2,
    }

    # Process each dataset
    for name, df in datasets.items():
        process_dataset(df, name, tokenizer, model, retriever)

    # Feature Extraction for machine learning
    file_paths = [
        "D:\Desktop\SI630\Project\RealNewsGuard\data\Retriever Dataset\RAG_results_test1.parquet",
        "D:\Desktop\SI630\Project\RealNewsGuard\data\Retriever Dataset\RAG_results_test2.parquet",
        "D:\Desktop\SI630\Project\RealNewsGuard\data\Retriever Dataset\RAG_results_train.parquet",
    ]
    extract_features_from_files(file_paths)


if __name__ == "__main__":
    main()
