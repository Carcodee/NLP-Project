import pandas as pd
import nltk 
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("comments_data.csv")

#data exploration
print("---------------Data exploration---------------")
print(df.head(5))
print(df.info())
print(df.describe())

print(df[['CONTENT', 'CLASS']])

df = df.drop(columns=['COMMENT_ID','AUTHOR','DATE'], axis=1)

print("---------------End data exploration---------------")

nltk.download('punkt')
nltk.download('stopwords')

vectorizer = CountVectorizer()
# hello, how are you, you , you?
# [hello , how, are, you]
# [tfidf, tfidf, tfidf, tfidf]

#tf(you) = 3/6 = 1/2
#idf(you) = log(350/250)

vectorized_matrix = vectorizer.fit_transform(df['CONTENT'])

#it makes a sparse matrix where it contains the frecuency of each word on every document (1 doc 1 row- 1 row have all words possible)
print("frecuency table shape:",vectorized_matrix.shape)
non_zero_cols = vectorized_matrix[0].indices
non_zero_cols_counts = vectorized_matrix[0].data;
non_zero_data = [(int(c), int(v)) for c, v in zip(non_zero_cols, non_zero_cols_counts)];
print("Non zero (idx, counts):", non_zero_data)
#features // tokens
print(vectorizer.get_feature_names_out())


#[hello, huh, out, !, xxx, ]

print('---------------------------------------------')
#it makes a sparse matrix where it stores all the tf-idf of each word 
vectorizer_tfidf = TfidfTransformer()
X = vectorizer_tfidf.fit_transform(vectorized_matrix)
y = df['CLASS'].values

print("TFIDF table shape:",vectorized_matrix.shape)
non_zero_cols = X[0].indices
non_zero_cols_tfidf = X[0].data;
non_zero_data = [(int(c), float(v)) for c, v in zip(non_zero_cols, non_zero_cols_tfidf)];
print("Non zero (idx, tfidf):", non_zero_data)


#features
print(vectorizer_tfidf.get_feature_names_out())

#creates an array with all row indices from 0-350 in this case
indices = np.arange(X.shape[0])
rng = np.random.default_rng(seed=15)
#randomize all the indices so if before was [0, 1, ..., 350] now is [3, 25, 8, ..., some index bewtween 0 and 350] so it becomes a random array of indices
rng.shuffle(indices)

#we create an array with remapped indices so now all rows are shuffled
X_shuffled = X[indices]
y_shuffled = y[indices]

train_size = int(0.75 * len(df))

#75 train 25 test with shuffled arrays
y_train = y_shuffled[:train_size]
y_test = y_shuffled[train_size:]

X_train = X_shuffled[:train_size]
X_test = X_shuffled[train_size:]

model = MultinomialNB().fit(X=X_train, y=y_train)

y_pred = model.predict(X=X_test)

#some trivial predictions
for truth, pred in zip(y_test[:10], y_pred[:10]):
    print(f"truth: {truth}, predicted: {pred}")

#do here the rest of validations n fold, cross val, accuracy, etc

# ===== CROSS VALIDATION SECTION =====
# Cross validation helps evaluate the model's performance more reliably
# It divides the training data into k folds and trains/tests the model k times,
# each time using a different fold as the test set and the remaining folds as training data.
# This should gives us a estimate of the model's performance on unseen data.

from sklearn.model_selection import cross_val_score

print("")
print("---------------5-Fold Cross Validation---------------")

# Initialize a fresh MultinomialNB model for cross-validation
# We create a new instance rather than using the previously trained model
cv_model = MultinomialNB()

# Perform 5-fold cross-validation on the training data
# cv=5: divides X_train into 5 folds
# scoring='accuracy': evaluates using accuracy metric (correct predictions / total predictions)
# cross_val_score automatically handles the train/test splits for each fold
cv_scores = cross_val_score(cv_model, X_train, y_train, cv=5, scoring='accuracy')

# Print the accuracy score for each individual fold
# This shows how consistent the model performs across different data splits
print("Cross-validation accuracy scores:", cv_scores)

# Print the mean (average) accuracy across all 5 folds
# This is our overall cross-validation accuracy metric
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

print("---------------End 5-Fold Cross Validation---------------")

# ===== CONFUSION MATRIX & ACCURACY =====
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print("")
print("\n---------------Model Evaluation on Test Set---------------")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

# Classification Report (Precision, Recall, F1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["NOT SPAM", "SPAM"]))

print("---------------End Model Evaluation---------------")

# ==========NEW COMMENTS CLASSIFICATION==========
print("")
print("\n---------------Classifying New Comments--------------")

# 6 new comments (4 non-spam, 2 spam)
new_comments = [
    "I really enjoyed this movie, the story was great!",
    "The film was enjoyable, I especially liked the soundtrack",
    "This movie was better than I expected, very emotional.",
    "Loved the cinematography, beautiful scenes!",
    "The movie had a nice balance of drama and humor.",
    "I enjoyed the plot, it was easy to follow and entertaining.",
    "The performances felt genuine, especially in the emotional scenes.",
    "I liked the ending, it wrapped up the story nicely.",
    "Click here to win a free iPhone!!!",
    "Subscribe to my channel and get free gifts now!",
]

# First: convert the comments into count vectors using SAME vectorizer
new_count_vectors = vectorizer.transform(new_comments)

# Then convert to TF-IDF using SAME transformer
new_tfidf = vectorizer_tfidf.transform(new_count_vectors)

# Predict using the trained model
predictions = model.predict(new_tfidf)

# Print the results clearly
for comment, pred in zip(new_comments, predictions):
    label = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"\nComment: {comment}\nPredicted class: {label}")

print("---------------End New Comments Classification---------------")
