# Linear-Algebra-for-AI---Course-Project
# Latent Semantic Analysis (LSA): A Linear Algebra Approach to Unsupervised Document Classification

**Author:** Avinash Sappati  
**Roll No:** 24AI10001  
**Course:** Linear Algebra for AI  
**Date:** December 1, 2025

## 📌 Project Overview
This project implements Latent Semantic Analysis (LSA) to automatically classify unstructured text documents into their respective topics without prior training (unsupervised learning). By utilizing Singular Value Decomposition (SVD), the project reduces the dimensionality of the feature space to reveal hidden "latent" semantic relationships between words and documents.

## 🎯 Objectives
* **Analyze Noise:** Identify and filter noise inherent in raw text data (e.g., stop words).
* **Mathematical Representation:** Construct a Term-Document matrix using TF-IDF weighting.
* **Dimensionality Reduction:** Apply Truncated SVD to approximate the original matrix with lower rank.
* **Visualization:** Visually demonstrate the separation of topics in the 2D latent semantic space.

## 📂 Dataset
The project utilizes a subset of the **20 Newsgroups** dataset, comprising 1,771 documents across three distinct categories:
1.  **sci.space:** Astronomy, space missions, physics.
2.  **rec.autos:** Automobiles, engines, driving.
3.  **comp.graphics:** Computer graphics, rendering, software.

## 🧮 Mathematical Foundations
The core of this project relies on **Singular Value Decomposition (SVD)**. We decompose the Term-Document matrix $C$ (weighted by TF-IDF) into three components:

$$C = U \Sigma V^T$$

* **$U$ (Left Singular Vectors):** Represents the relationship between documents and latent topics.
* **$\Sigma$ (Singular Values):** Represents the "strength" or variance of each latent topic.
* **$V^T$ (Right Singular Vectors):** Represents the relationship between words and latent topics.

By keeping only the top $k$ singular values (Truncated SVD), we construct the best rank-$k$ approximation ($C_k$) according to the Eckart-Young-Mirsky Theorem, effectively filtering linguistic noise.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:**
    * `scikit-learn`: For TF-IDF Vectorization, Truncated SVD, and Dataset loading.
    * `matplotlib` / `seaborn`: For 2D projection and visualization.
    * `numpy`: For numerical operations.

## 📊 Key Results
The project reduced the data to 2 dimensions ($k=2$) for visual inspection.

### Latent Concept Interpretation
* **Latent Concept 1 (General Context):** High weights for common terms like "edu", "com", "article".
* **Latent Concept 2 (Topic Discriminator):** * **Positive weights:** space, nasa, orbit (Science/Space).
    * **Negative weights:** car, engine, dealer (Recreation/Autos).

### Visualization
The 2D projection confirmed the efficacy of the linear algebra approach:
* **Distinct Clusters:** Documents naturally separated into clusters based on topic without labels.
* **Orthogonality:** The "Space" and "Autos" topics were separated primarily along the Y-axis (Latent Concept 2).
* **Overlap:** The "Graphics" cluster shared some technical terminology overlap but maintained a distinct center of gravity.

## 🚀 Usage
1.  **Data Loading:** Fetch the 20 Newsgroups subset.
2.  **Preprocessing:** Apply TF-IDF with stop-word filtering (top 2000 features).
3.  **SVD:** Apply `TruncatedSVD` with $k=2$.
4.  **Analysis:** Extract top terms for each component and plot the $U$ matrix.
