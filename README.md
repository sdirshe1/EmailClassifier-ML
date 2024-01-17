# EmailClassifier-ML
# Overview:
EmailClassifier-ML is an advanced machine learning project aimed at classifying emails into spam and non-spam categories using Natural Language Processing (NLP) and Multinomial Naive Bayes classifiers. This project showcases the power of ML in automating email filtering processes.

# Key Components
Email Preprocessing: Utilizes regular expressions and NLTK for cleaning and preparing email data.
```data_cleaned['processed_message'] = data_cleaned['message'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())```

Feature Extraction: Applies CountVectorizer for converting email text into a machine-readable numeric format.
```vectorizer = CountVectorizer()```

Model Training & Evaluation: Employs MultinomialNB for model training and evaluation, providing metrics like accuracy and confusion matrix.
```nb_classifier = MultinomialNB()```

Visualization: Generates insightful plots such as distribution of spam vs non-spam emails and top terms in emails using Matplotlib and Seaborn.
```sns.barplot(x='Frequency', y='Term', data=top_terms)```

Dynamic Update Function: Allows real-time model updating with new email data.
```classify_and_update(new_email, actual_label)```
