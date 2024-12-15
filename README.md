# Network-Attack-Detection

## Table of Contents

- [Description](#description)
- [Key Features](#key-features)
- [Running the Program](#running-the-program)
- [Team Members and Responsibilities](#team-members-and-responsibilities)

---

## Description
Network-Analysis-Detection is a Jupyter Notebook-based project focused on detecting and analyzing network attack categories using datasets from Cyber Range Lab UNSW Canberra. The notebook includes the full end-to-end process of building machine learning models, from data preprocessing, handling imbalance, and training models to evaluation and visualization.

This project implements:

- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes (GNB)
- Decision Tree (ID3)

The task involves balancing imbalanced classes, optimizing feature scaling and encoding, and comparing from-scratch implementations with library-based models.

---

## Key Features

1. Data Preprocessing
    - Feature Scaling using StandardScaler
    - Feature Encoding using Label Encoding
    - Handling Class Imbalance with SMOTE
2. Model Implementation
    - From-Scratch and Library-Based Models (KNN, GNB, ID3)
3. Evaluation Metrics
    - Confusion Matrix
    - Macro-Average F1 Score
    - ROC-AUC Curve
4. Visualization
    - Model performance visualizations for better understanding of class predictions.

---

## Running the Program

1. Clone the repository
    ```bash
    git clone https://github.com/novelxv/Network-Analysis-Detection.git
    cd Network-Analysis-Detection
    ```
2. Install Dependencies

    Ensure you have `Python 3.8+` installed.

3. Run the Notebook

    Open the `Network-Analysis-Detection.ipynb` file and run the cells.

---

## Team Members and Responsibilities

**Group 19: Natural Intelligence**

| Name              | NIM          | Task Description                                                                                                                                             |
|-------------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Nabila Shikoofa Muida**    | 13522069      | EDA, Library Implementations, Document                                                                                    |
| **Novelya Putri Ramadhani**    | 13522096      | Data Preprocessing, KNN From-Scratch Implementations, Validation, Document
| **Hayya Zuhailii Kinasih**    | 13522102      | Data Preprocessing, ID3 From-Scratch Implementations, Document
| **Diana Tri Handayani**    | 13522104      | EDA, GNB From-Scratch Implementations, Document

---