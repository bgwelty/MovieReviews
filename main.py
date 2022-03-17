import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# read in data
review_df = pd.read_csv('IMDB Dataset.csv')

# sample 1000 negative reviews and 9000 positive
negative_review_df = review_df[review_df['sentiment'] == 'negative'][:1000]
positive_review_df = review_df[review_df['sentiment'] == 'positive'][:9000]

# create imbalanced dataframe
imbalanced_review_df = pd.concat([positive_review_df, negative_review_df])

# randomly undersample the positive reviews to create a balanced sample (1000 if each sentiment)
rus = RandomUnderSampler(random_state=0)
balanced_review_df, balanced_review_df['sentiment'] = rus.fit_resample(imbalanced_review_df[['review']],
                                                                       imbalanced_review_df['sentiment'])

# split into train and test sets
train, test = train_test_split(balanced_review_df, test_size=.33, random_state=42)

# x is the independent variable (reviews), y is the dependent variable (sentiment)
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

# turn text into numerical vectors using bag of words: tf-idf
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)

# Fitting different types of models
# Support Vector Machine
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

# Decision Tree
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

# Naive Bayes
gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

# Evaluate Model Accuracies
model_accuracies = {'SVM': 0, 'Decision Tree': 0, 'Naive Bayes': 0, 'Logistic Regression': 0}
model_accuracies['SVM'] = svc.score(test_x_vector, test_y)
model_accuracies['Decision Tree'] = dec_tree.score(test_x_vector, test_y)
model_accuracies['Naive Bayes'] = gnb.score(test_x_vector.toarray(), test_y)
model_accuracies['Logistic Regression'] = log_reg.score(test_x_vector, test_y)
# SVM is most accurate with ~0.84 accuracy
for model in model_accuracies:
    print(model, ':', model_accuracies[model].round(2))
    print()

# compute F1 scores for SVM only
f1_score(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'], average=None)

# build classification report for SVM
print('Support Vector Machine classification report:')
print(classification_report(test_y, svc.predict(test_x_vector), labels=['positive', 'negative']))

# Confusion Matrix for SVM
conf_mat = confusion_matrix(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'])
print(conf_mat)
print('290 True positives, 45 False positives, 60 False negatives, 265 True negatives')
print()

# Optimizing model by adjusting parameters
parameters = {'C': [1, 4, 8, 16, 32], 'kernel': ['linear', 'rbf']}
svc = SVC()
svc_grid = GridSearchCV(svc, parameters, cv=5)
svc_grid.fit(train_x_vector, train_y)

print(svc_grid.best_params_)
print(svc_grid.best_estimator_)
print('The model was already using the optimal parameters: C=1, kernel=\'linear\'')
