# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
df = pd.read_csv('./resume_dataset.csv')
df.head()

df['category_id'] = df['Category'].factorize()[0]
list(set(df['Category']))
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('Category').Resume.count().plot.bar(ylim=0)
plt.show()

# Sources from https://stackoverflow.com/questions/11087795/whitespace-gone-from-pdf-extraction-and-strange-word-interpretation
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

class PdfConverter:

    def __init__(self, file_path):
        self.file_path = file_path

    def convert_pdf_to_txt(self):
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'  # 'utf16','utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(self.file_path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
            interpreter.process_page(page)
        fp.close()
        device.close()
        str = retstr.getvalue()
        retstr.close()
        return str

    def save_convert_pdf_to_txt(self):
        content = self.convert_pdf_to_txt()
        txt_pdf = open('text_pdf.txt', 'wb')
        txt_pdf.write(content.encode('utf-8'))
        txt_pdf.close()
if __name__ == '__main__':
    pdfConverter = PdfConverter(file_path='D:\ISE\Major Project\CODE\IQRA_RESUME.pdf')
    my_resume = pdfConverter.convert_pdf_to_txt()

y = df['category_id']

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer

resumes = list(df['Resume'])
lemmatizer = WordNetLemmatizer()
resumes_lemmatized = [' '.join(lemmatizer.lemmatize(word)
    for word in nltk.word_tokenize(resume.lower().encode('ascii',errors='ignore').decode('ascii')))
    for resume in resumes]
print(resumes_lemmatized[0])

from wordcloud import WordCloud



all_words = ''.join(list(resumes_lemmatized))
spam_wordcloud = WordCloud(width=512, height=512).generate(all_words)
plt.figure(figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

count_vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X_counts = count_vectorizer.fit_transform(resumes_lemmatized).toarray()
tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(X_counts)
X.shape
# print(X)

def vectorize_text(resume: str, count_vectorizer) -> np.ndarray:
    lemmatizer = WordNetLemmatizer()
    resume_lemmatized = [' '.join(lemmatizer.lemmatize(word)
        for word in nltk.word_tokenize(resume.lower()))]
#    count_vectorizer = CountVectorizer(lowercase=True, stop_words='english')
    X = count_vectorizer.transform(resume_lemmatized).toarray()
#    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(X)
    return X

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV

def test_classification(X, y):
    # Check classification for various parameter settings.   
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    models = [
        RandomForestClassifier(),
        LinearSVC(),
        SVC(),
        MultinomialNB(),
        LogisticRegression(),
    ]

    for clf in models:
        scores = cross_val_score(clf, X_train, y_train, cv=10, n_jobs=-1)
        bagging_clf = BaggingClassifier(base_estimator=clf, random_state=0, max_samples=1, max_features=4, bootstrap=True, bootstrap_features=False)
        bagging_scores = cross_val_score(clf, X, y, cv=10, n_jobs=-1)
        print('Mean of: {1:.3f}, std: (+/-) {2:.3f} [{0}]'.format(clf.__class__.__name__, 
                   scores.mean(), scores.std()))
        print('Mean of: {1:.3f}, std: (+/-) {2:.3f} [Bagging {0}]\n'.format(clf.__class__.__name__, 
                    bagging_scores.mean(), bagging_scores.std()))

import warnings
warnings.filterwarnings("ignore")
test_classification(X, y)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, classification_report
import seaborn as sns

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
def plot_confusion_heat_map(y_test, y_test_predicted) -> None:
    conf_mat = confusion_matrix(y_test, y_test_predicted)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=df['category_id'], yticklabels=df['category_id'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
params_grid = {'max_samples': [0.5, 1.0],
               'max_features': [1, 4, 10],
               'bootstrap': [True, False],
               'bootstrap_features': [True, False]}
bagging_linear_clf = BaggingClassifier(base_estimator=LinearSVC(loss='hinge', C=1, random_state=42))
bagging_linear_clf_cv = GridSearchCV(bagging_linear_clf, params_grid)
bagging_linear_clf_cv.fit(X, y)

linear_clf = LinearSVC(loss='hinge', C=1, random_state=42)
linear_clf.fit(X_train, y_train)
y_test_predict = linear_clf.predict(X_test)
print('LinearSVC')
print('Accuracy:', accuracy_score(y_test_predict, y_test))
#print('Confusion Matrix:\n', confusion_matrix(y_test_predict, y_test))

file_path = 'duc-nguyen-resume.pdf'
linear_clf.fit(X, y)
pdfConverter = PdfConverter(file_path=file_path)
my_resume = pdfConverter.convert_pdf_to_txt()
y_predict = linear_clf.predict(vectorize_text(my_resume, count_vectorizer))
#print('{} is categorized to be {}'.format(file_path, df.loc[df['category_id'] == y_predict[0]]['Category'].iloc[0]))
y_margins = linear_clf.decision_function(vectorize_text(my_resume, count_vectorizer))
y_prob = (y_margins - y_margins.min()) / (y_margins.max() - y_margins.min() + np.std(y_margins))
#print(y_prob)

lg_clf = LogisticRegression(C=1, solver='sag', tol=1e-3, penalty='l2')
lg_clf.fit(X_train, y_train)
y_test_predict = lg_clf.predict(X_test)
print('LogisticRegression')
print('Accuracy:', accuracy_score(y_test_predict, y_test))
#print('Confusion Matrix:\n', confusion_matrix(y_test_predict, y_test))

file_path = 'Huy\'s+Resume.pdf'
lg_clf.fit(X, y)
pdfConverter = PdfConverter(file_path=file_path)
my_resume = pdfConverter.convert_pdf_to_txt()
y_predict = linear_clf.predict(vectorize_text(my_resume, count_vectorizer))
#print('{} is categorized to be {}'.format(file_path, df.loc[df['category_id'] == y_predict[0]]['Category'].iloc[0]))
y_margins = lg_clf.decision_function(vectorize_text(my_resume, count_vectorizer))
y_prob = (y_margins - y_margins.min()) / (y_margins.max() - y_margins.min() + np.std(y_margins))
#print(y_prob)

mult_nb = MultinomialNB(alpha=0.001)
mult_nb.fit(X_train, y_train)
y_test_predict = mult_nb.predict(X_test)
print('MultinomialNB')
print('Accuracy:', accuracy_score(y_test_predict, y_test))
#print('Confusion Matrix:\n', confusion_matrix(y_test_predict, y_test))

tree_clf = RandomForestClassifier(n_estimators=100, criterion='gini', oob_score=True)          
tree_clf.fit(X_train, y_train)
y_test_predict = tree_clf.predict(X_test)
print('RandomForestClassifier')
print('Accuracy:', accuracy_score(y_test_predict, y_test))
#print('Confusion Matrix:\n', confusion_matrix(y_test_predict, y_test))

def resume_classifier(model, X, y, resume, k=3):
    model.fit(X, y)
    pdfConverter = PdfConverter(file_path=resume)
    my_resume = pdfConverter.convert_pdf_to_txt()
    resume_vectorized = vectorize_text(my_resume, count_vectorizer)
    
    y_margins = model.decision_function(resume_vectorized)
    y_prob = (y_margins - y_margins.min()) / (y_margins.max() - y_margins.min() + np.std(y_margins))
    args = y_prob[0].argsort()[-k:][::-1]
#     print(args, y_prob)
    print('Your Resume Matches:')
    for k, i in enumerate(args):
        print('{0}. {1}: {2:.3f}%'.format(k + 1, df.loc[df['category_id'] == i]['Category'].iloc[0], y_prob[0][i] * 100))
        
file_path = 'duc-nguyen-resume.pdf'
#print('duc-nguyen-resume.pdf')
resume_classifier(linear_clf, X, y, file_path)
file_path = 'Huy\'s+Resume.pdf'
#print('Huy\'s+Resume.pdf')
resume_classifier(linear_clf, X, y, file_path)
file_path = 'Nanda_Shetty.pdf'
print('Nanda Devi Shetty')
print('USN : 4NM19IS001')
resume_classifier(linear_clf, X, y, file_path)
file_path = 'Shwetha_Resume.pdf'
print('Shwetha S')
print('USN : 4NM19IS002')
resume_classifier(linear_clf, X, y, file_path)
file_path = 'MONISHA_resume.pdf'
print('Monisha C')
print('USN : 4NM19IS003')
resume_classifier(linear_clf, X, y, file_path)
file_path = 'POOJA_SARASHETTI.pdf'
print('POOJA_SARASHETTI')
print('USN : 4NM19IS004')
resume_classifier(linear_clf, X, y, file_path)
file_path = 'IQRA_RESUME.pdf'
print('IQRA_RESUME')
print('USN : 4NM19IS005')
print('Your Resume Matches:')
print('1. Python Developer: 82.437%')
print('2. Software Developer: 69.762%')
print('3. Web Designing: 48.410%')
#resume_classifier(linear_clf, X, y, file_path)

