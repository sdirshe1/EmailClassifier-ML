from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib
import pandas as pd  
import nltk
import seaborn as sns
from scipy.sparse import vstack
from sklearn.naive_bayes import MultinomialNB  
import numpy as np
import re  # For regular expression operations
from sklearn.feature_extraction.text import CountVectorizer  
from nltk.corpus import stopwords
import matplotlib.pyplot as plt  
nltk.download('stopwords')
matplotlib.use('Agg')  

# Load the dataset
emails_filepath = 'messages.csv'
emails_df = pd.read_csv(emails_filepath)

cleaned_emails_df = emails_df.dropna().copy()

stop_words = set(stopwords.words('english'))
cleaned_emails_df['clean_subject'] = cleaned_emails_df['subject'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())
cleaned_emails_df['clean_message'] = cleaned_emails_df['message'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())
cleaned_emails_df['clean_subject'] = cleaned_emails_df['clean_subject'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
cleaned_emails_df['clean_message'] = cleaned_emails_df['clean_message'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# Save the cleaned data to a CSV file
cleaned_data_filepath = 'cleaned_emails.csv'
cleaned_emails_df.to_csv(cleaned_data_filepath, index=False)

vectorizer = CountVectorizer()
features_matrix = vectorizer.fit_transform(cleaned_emails_df['clean_message'])


X_train, X_test, y_train, y_test = train_test_split(features_matrix, cleaned_emails_df['label'], test_size=0.3, random_state=42)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

predictions = nb_classifier.predict(X_test)

confusion_mtx = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap="Blues")
plt.title('Confusion Matrix for Email Classification')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
confusion_mtx_filepath = "email_confusion_mtx.png"
plt.savefig(confusion_mtx_filepath, dpi=300)
plt.close()

classification_accuracy = metrics.accuracy_score(y_test, predictions)
print(f"Classification Accuracy: {classification_accuracy:.2%}")
classification_report_filepath = "email_classification_report.txt"
with open(classification_report_filepath, "w") as report_file:
    class_report = metrics.classification_report(y_test, predictions)
    report_file.write(class_report)
    report_file.write("\nConfusion Matrix:\n")
    report_file.write(str(confusion_mtx))

email_labels = ['Non-Spam', 'Spam']
email_counts = cleaned_emails_df['label'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(email_counts, labels=email_labels, colors=['skyblue', 'orange'], autopct='%1.1f%%', startangle=140, explode=(0.1, 0))
plt.title('Distribution of Spam vs Non-Spam Emails')
email_distribution_filepath = "email_distribution_pie.png"
plt.savefig(email_distribution_filepath, dpi=300)
plt.close()

def visualize_top_terms(text_series, chart_title, num_terms=20, chart_filename="top_terms_visual.png"):
    # Generate term frequency matrix
    vectorizer = CountVectorizer(stop_words='english')
    term_matrix = vectorizer.fit_transform(text_series)
    term_sum = term_matrix.sum(axis=0)
    term_freqs = [(word, term_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    term_freqs.sort(key=lambda x: x[1], reverse=True)
    top_terms = pd.DataFrame(term_freqs[:num_terms], columns=['Term', 'Frequency'])

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Term', data=top_terms)
    plt.title(chart_title)
    plt.xlabel('Occurrence Count')
    plt.ylabel('Significant Terms')
    plt.savefig(chart_filename, dpi=300)
    plt.close()

visualize_top_terms(cleaned_emails_df[cleaned_emails_df['label'] == 1]['clean_message'], 'Frequent Terms in Spam Emails', chart_filename="spam_terms_visual.png")
visualize_top_terms(cleaned_emails_df[cleaned_emails_df['label'] == 0]['clean_message'], 'Frequent Terms in Non-Spam Emails', chart_filename="non_spam_terms_visual.png")

file_path = 'messages.csv'  
data = pd.read_csv(file_path)  

data_cleaned = data.dropna().copy() 

stop_words_set = set(stopwords.words('english')) 
data_cleaned['processed_subject'] = data_cleaned['subject'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())
data_cleaned['processed_message'] = data_cleaned['message'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())
data_cleaned['processed_subject'] = data_cleaned['processed_subject'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words_set))
data_cleaned['processed_message'] = data_cleaned['processed_message'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words_set))

# Save the processed data to a new CSV file
processed_file_path = 'processed_messages.csv'  
data_cleaned.to_csv(processed_file_path, index=False) 

# Distribution of Spam vs Non-Spam Messages
sns.set(style="whitegrid")  
plt.figure(figsize=(8, 5))  
spam_dist_plot = sns.countplot(x='label', data=data_cleaned)  
plt.title('Distribution of Spam vs Non-Spam Messages')  
plt.xlabel('Email Type') 
plt.ylabel('Count') 
spam_dist_plot.figure.savefig("spam_distribution.png") 

# Top Terms Visualization
def plot_top_terms(text, title, n_terms=20, filename="top_terms.png"):
    vect = CountVectorizer(stop_words='english') 
    term_matrix = vect.fit_transform(text)  
    term_sum = term_matrix.sum(axis=0) 
    term_freq = [(word, term_sum[0, idx]) for word, idx in vect.vocabulary_.items()]  
    term_freq = sorted(term_freq, key=lambda x: x[1], reverse=True)[:n_terms] 
    
    top_df = pd.DataFrame(term_freq, columns=['Term', 'Frequency'])
    
    # Plot
    plt.figure(figsize=(10, 6))  # Setting the figure size for the plot
    terms_plot = sns.barplot(x='Frequency', y='Term', data=top_df) 
    plt.title(title)  
    plt.xlabel('Frequency')  
    plt.ylabel('Top Terms')  
    terms_plot.figure.savefig(filename)  
plt.savefig('plot_173.png')

spam_text = data_cleaned[data_cleaned['label'] == 1]['processed_message']  
plot_top_terms(spam_text, 'Top Terms in Spam Messages', filename="spam_top_terms.png")  # Plotting and saving top terms in spam messages

# Visualize and save top terms in non-spam messages
non_spam_text = data_cleaned[data_cleaned['label'] == 0]['processed_message']  
plot_top_terms(non_spam_text, 'Top Terms in Non-Spam Messages', filename="non_spam_top_terms.png")  

# Feature Extraction - Word Frequencies
vectorizer = CountVectorizer() 
X_features = vectorizer.fit_transform(data_cleaned['processed_message'])

# Splitting Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_features, data_cleaned['label'], test_size=0.3, random_state=42)  

# Model Training
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Model Evaluation
y_pred = nb_classifier.predict(X_test) 

# Print metrics
print(metrics.classification_report(y_test, y_pred)) 
print(metrics.confusion_matrix(y_test, y_pred))  
accuracy = metrics.accuracy_score(y_test, y_pred)  
print(f"Accuracy: {accuracy:.2%}") 

# Save metrics to a file
with open("classification_report.txt", "w") as f:  
    f.write(metrics.classification_report(y_test, y_pred)) 
    f.write("\nConfusion Matrix:\n")  
    f.write(str(metrics.confusion_matrix(y_test, y_pred)))  
    f.write(f"\nAccuracy: {accuracy:.2%}")  

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    text = ' '.join(word for word in text.split() if word not in stop_words_set)
    return text

# Initialize the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()

# Train the model using the training data
nb_classifier.fit(X_train, y_train)

#Evaluate the model's performance on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate and print the accuracy
initial_accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Initial Accuracy: {initial_accuracy:.2%}")

# Print the classification report
print(metrics.classification_report(y_test, y_pred))

# Print the confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('plot_245.png')

def classify_and_update(new_email, actual_label):
    # Preprocess and vectorize the new email
    new_email_processed = preprocess_text(new_email)
    new_email_vectorized = vectorizer.transform([new_email_processed])

    # Classify the new email
    new_email_prediction = nb_classifier.predict(new_email_vectorized)
    print("The new email is classified as:", "Spam" if new_email_prediction[0] == 1 else "Ham")

    # Update the model with the new data
    global X_train, y_train
    X_train = vstack([X_train, new_email_vectorized])
    y_train = np.append(y_train, actual_label)
    nb_classifier.fit(X_train, y_train)

    # Retest the model on the test set
    y_pred_updated = nb_classifier.predict(X_test)
    updated_accuracy = metrics.accuracy_score(y_test, y_pred_updated)
    print(f"Updated Accuracy: {updated_accuracy:.2%}")

# BONUS
new_email_input = input("Enter a new email to classify: ")
new_email_label = int(input("Enter the true label (1 for Spam, 0 for Ham): "))
classify_and_update(preprocess_text(new_email_input), new_email_label)

# Adjust the trade-off between false positives and false negatives
def adjust_threshold(threshold=0.5):
    y_probs = nb_classifier.predict_proba(X_test)[:, 1]
    y_pred_adjusted = (y_probs >= threshold).astype(int)
    adjusted_accuracy = metrics.accuracy_score(y_test, y_pred_adjusted)
    print(f"Adjusted Accuracy with Threshold {threshold}: {adjusted_accuracy:.2%}")

threshold_value = float(input("Enter a new threshold (between 0 and 1) for classifying spam: "))
adjust_threshold(threshold_value)
