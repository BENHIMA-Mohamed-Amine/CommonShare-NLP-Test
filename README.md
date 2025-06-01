# Report CommonShare Assignment

## Advanced NLP (Natural Language Processing)

**By:** BENHIMA Mohemed-amine

---

[![Open the notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QwPhhWu25rzze4BEUcrv5qtR02U3SsGF#scrollTo=Usalmh189VmU)


## Contents

- I. Sentiment Analysis  
  1. Methodology  
  2. Challenges faced  
  3. Results  
  4. Technologies  

- II. Topic Modeling  
  1. Methodology  
  2. Challenges faced  
  3. Results  
  4. Technologies  

- III. Named Entity Recognition (NER)  
  1. Methodology  
  2. Challenges faced  
  3. Results  
  4. Technologies  

- IV. Text Summarization  
  1. Methodology  
  2. Challenges faced  
  3. Results  
  4. Technologies  

---

# I. Sentiment Analysis

### Methodology

I fine-tuned DistillBERT on a subset of (5000 rows) the Twitter Sentiment Analysis dataset in Kaggle.

**Steps:**

- Loading the Data (3 labels: negative, neutral, positive)  
- Cleaning the Data for text classification task  
  - Remove duplicates  
  - Remove nulls  
  - Remove unused columns  
  - Remove URLs  
  - Remove html tags  
  - Handle spaces  
  - Convert to lowercase  
- Tokenization  
  - Using padding and truncation  
  - Fix the sentence to the default max_length = 512  
- Define evaluation metrics:  
  - Accuracy  
  - Recall ⇒ macro (average of all classes recalls)  
  - Precision ⇒ macro  
  - F1 ⇒ macro  
- Define Training Arguments:  
  - L2 regularization = 0.01  
  - Train and Validation batch size = 16  
  - Learning rate scheduler: Linear for faster convergence  
  - Warmup is 20% of total steps (batches)  
- Load at the end the best model with the best F1 score  
- Use half-precision (fp16) for increasing speed training and memory efficiency  
- Other hyperparameters were left as default  

**Training:**  
For 5 epochs (due to compute constraints)

### Challenges faced

- Compute constraints (used up Colab and close to Kaggle quota)  
- Struggling to find a real-world dataset

### Results

- Accuracy, precision, recall, and F1 are around 78%  
- For 5 epochs training on 5000 samples, it’s a good result  
- Using a more complex model like BERT-base can improve results  
- Cleaning the dataset more can help improve results  
- Training for more epochs can increase results

### Technologies

- pandas  
- HuggingFace Ecosystem  
- Transformers  
- Datasets  
- Evaluate  
- HuggingFace Hub  
- ClearML  
- Kaggle

---

# II. Topic Modeling

### Methodology

I used Latent Dirichlet Allocation (LDA) for topic modeling.

**Steps:**

- Pre-process the dataset:  
  - Remove URLs  
  - Remove mentions  
  - Remove hashtags  
  - Remove punctuation  
  - Lowercase  
  - Lemmatization  
- Convert the data to a Bag of Words corpus  
- Define hyper-parameter ranges  
- Apply grid search to find best hyper-parameters  
- Use Coherence to select best combination  
- Train LDA with best hyper-parameters  
- Display some topics  
- Visualize using pyLDAvis  

### Challenges faced

- Twitter dataset is messy, so pre-processing carefully was important

### Results

- Best coherence score was 40%, which is good for a messy Twitter dataset

### Technologies

- NLTK  
- SpaCy  
- Gensim  
- pyLDAvis

---

# III. Named Entity Recognition (NER)

### Methodology

I used SpaCy pre-trained NER on the Twitter Sentiment Analysis dataset.

### Challenges faced

- No challenges

### Results

- Without training, results are good.  
- Fine-tuning with a transformer-based NER like BERT can improve results.  
- Overall, no training still yields good results.

### Technologies

- SpaCy

---

# IV. Text Summarization

### Methodology

- Used a BERT-based model fine-tuned for extractive summarization.  
- Each sentence represented by embedding of its first token (CLS token).  
- Added a simple linear classifier on top of BERT to predict sentence importance scores.  
- No training or fine-tuning done; used model as-is.

### Challenges faced

- Loading pre-trained weights properly without fine-tuning.  
- HuggingFace does not support a pipeline for extractive summarization.

### Results

- Without fine-tuning, very good results obtained.

### Technologies

- HuggingFace BERT  
- NLTK
