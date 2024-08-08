import re
import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer



class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self._preprocess(text) for text in X]
        
    def _preprocess(self, text):
        lemmatizer = WordNetLemmatizer()
        text= text.lower().strip()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
        text = ' '.join(text)  
        """url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text = url_pattern.sub(r'', text)
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        
        html_tags_pattern = r'<.*?>'

        text = re.sub(html_tags_pattern, '', text)
        text= re.sub(r'\s+', ' ', text)
        text= text.lower().strip()"""
        return ' '.join(text)

data = pd.read_csv("data\\train\\train.csv",encoding="utf-8",low_memory=False)
print(data)
data_ = data['text']
calsses = data['classification']

train_tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b(?!\d{1,})(?=\w{2,})\w+\b', stop_words='english', lowercase=True, max_features=2000)

train_tfidf_vectorizer.fit_transform(data_, calsses)

clf_pipeline = Pipeline([
                ('preprocessor', TextPreprocessor()),
                ('tfidf', train_tfidf_vectorizer),
                ('select_best', SelectKBest(chi2, k=200)),
            ])

print("fitting the vectorizer")
clf_pipeline.fit(data_, calsses)
            

print(clf_pipeline['tfidf'].get_feature_names_out())
words = clf_pipeline['tfidf'].get_feature_names_out()

print(list(words))
print("-------------------------------------------------")
print(words[clf_pipeline['select_best'].get_support()])
joblib.dump(clf_pipeline, "models\\tf_idf_model.pkl")