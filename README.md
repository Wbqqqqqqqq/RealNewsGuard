# README

## Project Overview
This project focuses on detecting AI-generated fake news using a knowledge-augmented retrieval framework. It combines retrieval-based techniques with advanced language models to improve accuracy.

## Code Structure
```
Project Root/
│── assets/
│   ├── LoRA/
│   ├── LoRA+RAG/
│   ├── Retriever/
│── data/
│   ├── Dataset/
│   ├── Retreiver Dataset/
│── evaluate/
│   ├── result/
│   ├── evaluate.ipynb
│── prepare/
│   ├── data_fetch_main_dataset.ipynb
│   ├── data_fetch_retriever_dataset.ipynb
│   ├── data_generation.py
│   ├── retreiver_dataset_construction.py
│   ├── retreiver.py
│── train/
│   ├── lora_fine_tune_with_retriever.ipynb
│   ├── machine_learning_baseline.ipynb
│── infer.py
│── requirement.txt
│── README.md
```

## **Installation**
Ensure you have Python installed, then install dependencies with:

```shell
pip install -r requirements.txt
```

## **Usage**

### **Running Inference**
Use the `infer.py` script to check if a given text is real or fake news.

#### **Example 1: Fake News Detection**
```shell
python infer.py --text "In a groundbreaking expedition to the uncharted regions of the Peruvian Amazon, a team of scientists from the Natural History Museum of London has made the discovery of a lifetime, uncovering 27 previously unknown species of plants and animals that are set to rewrite the textbooks on biodiversity."
```

#### **Example 2: Real News Detection**
```shell
python infer.py --text "Senator Mark Warner, who is the vice-chair of the Senate intel committee, called the Trump administration’s decision to fire top FBI officials “deeply alarming”."
```
