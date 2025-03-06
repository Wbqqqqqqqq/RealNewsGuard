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
│── prepare/
│   ├── data_fetch_main_dataset.ipynb
│   ├── data_fetch_retriever_dataset.ipynb
│   ├── data_generation.py
│   ├── retreiver_dataset_construction.py
│   ├── retreiver.py
│── train/
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
python infer.py --text "In a groundbreaking discovery, scientists at the International Coffee Research Institute (ICRI) have found a rare coffee plant in the high-altitude regions of Colombia that naturally produces espresso beans." --checkpoint_path "assets/LoRA+RAG"
```

#### **Example 2: Real News Detection**
```shell
python infer.py --text "Senator Mark Warner, who is the vice-chair of the Senate intel committee, called the Trump administration’s decision to fire top FBI officials “deeply alarming”." --checkpoint_path "assets/LoRA+RAG"
```
