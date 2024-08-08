
import re
import numpy as np
import pandas as pd
import langid
from sklearn.calibration import LabelEncoder

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
import joblib
import datetime
from datetime import datetime

from utils.pdf_module import *

import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

current_hour = datetime.now()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    This class is an sklearn Pipeline Layer that is responsible for text preprocessing, including tokenization, stopword removal,
    lemmatization, stemming, and removal of URLs, numbers, and emojis.

    Args:
        BaseEstimator (sklearn.base.BaseEstimator): Base class for all estimators in scikit-learn.
        TransformerMixin (sklearn.base.TransformerMixin): Mixin class for all transformers in scikit-learn.
    """
    def __init__(self, n_jobs=-1):
        """
        Initializes the text preprocessor with stopwords, lemmatizer, stemmer, and patterns for emojis, URLs, and HTML tags

        Args:
            n_jobs (int, optional): The number of jobs to run in parallel. Default is -1, which means using all processors.
        """
        #Set of English stopwords.
        self.stop_words = set(stopwords.words('english'))
        #Lemmatizer object for word lemmatization.
        self.lemmatizer = WordNetLemmatizer() 
        #Stemmer object for word stemming.
        self.stemmer = PorterStemmer() 
        #Number of parallel jobs for preprocessing
        self.n_jobs = n_jobs 
        #Regular expression pattern for emojis.
        self.emoji_pattern = re.compile(r'^(?:[\u2700-\u27bf]|(?:\ud83c[\udde6-\uddff]){1,2}|(?:\ud83d[\udc00-\ude4f]){1,2}|[\ud800-\udbff][\udc00-\udfff]|[\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e]|\u3299|\u3297|\u303d|\u3030|\u24c2|\ud83c[\udd70-\udd71]|\ud83c[\udd7e-\udd7f]|\ud83c\udd8e|\ud83c[\udd91-\udd9a]|\ud83c[\udde6-\uddff]|\ud83c[\ude01-\ude02]|\ud83c\ude1a|\ud83c\ude2f|\ud83c[\ude32-\ude3a]|\ud83c[\ude50-\ude51]|\u203c|\u2049|\u25aa|\u25ab|\u25b6|\u25c0|\u25fb|\u25fc|\u25fd|\u25fe|\u2600|\u2601|\u260e|\u2611|[^\u0000-\u007F])+$')
        #Regular expression pattern for URLs
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        #Regular expression pattern for HTML tags
        self.html_tags_pattern = re.compile(r'<.*?>')

    #Fits the transformer; does nothing in this case.
    def fit(self, X, y=None):
        """
        This method doesn't perform any fitting but is required by scikit-learn's transformer interface
       
        Args:
            X (array-like): Input data.
            y (array-like, optional): Target data.

        Returns:
            _type_: self
        """
        return self

    #Applies preprocessing to the input text
    def transform(self, X, y=None):
        """
        Applies the _preprocess method to each element in X using parallel processing.
        Args:
            X (array-like): Input data.
            y (array-like, optional): Target data.

        Returns:
            _type_: list of str
        """
        return joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(self._preprocess)(text) for text in X)

    #Internal method to preprocess a single text string
    def _preprocess(self, text):
        """
        Cleans and preprocesses the input text by removing URLs, numbers, punctuation, emojis,
        and HTML tags, and by applying lemmatization and stopword removal.

        Args:
            text (str): Input text

        Returns:
            _type_: str
        """
        text = text.lower().strip()
        #lemmatizing the text to original form
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split() if word not in self.stop_words])
        text = self.url_pattern.sub(r'', text)
        #remove numbers
        text = re.sub(r'[0-9]+', '', text)
        #remove anything but words and spaces
        text = re.sub(r'[^\w\s]', '', text)
        #remove emojis
        text = self.emoji_pattern.sub('', text)
        #remove html tags
        text = self.html_tags_pattern.sub('', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

class TrainingModels:
    """
        This class handles the training and evaluation of multiple classifiers on preprocessed text data
    """
    def __init__(self):
        """
        Initializes the training models, including setting up the TF-IDF vectorizer and classifiers
        """
        #TF-IDF Vectorizer object for text vectorization.
        self.train_tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b(?!\d{1,})(?=\w{2,})\w+\b', stop_words='english', lowercase=True, max_features=constants.feature_number)
        #List of classifier tuples (name, classifier).
        self.classifiers = [
            #('XGB',xgb.XGBClassifier(verbosity=1,n_estimators=constant.n_estimators,max_depth=30,subsample=0.6,eval_metric='mlogloss',objective='multi:softmax')),
            ('RandomForest', RandomForestClassifier(random_state=42,n_estimators=constants.n_estimators,max_depth=100,min_samples_split=35,min_samples_leaf=15,max_features='sqrt',class_weight=constants.class_weights)),
        ]
        self.env_classifiers = [
            #('XGB',xgb.XGBClassifier(verbosity=1,n_estimators=constant.n_estimators,max_depth=30,subsample=0.6,eval_metric='mlogloss',objective='multi:softmax')),
            ('RandomForest', RandomForestClassifier(random_state=42,n_estimators=constants.n_estimators,max_depth=40,min_samples_split=15,min_samples_leaf=5,max_features='sqrt')),
        ]
        #DataFrame for storing training data
        self.data = pd.DataFrame()
        #Label encoder for target labels.
        self.label_encoder = LabelEncoder()
        #Parameters for TF-IDF vectorization.
        self.tfidf_Params = []
        #Dictionary to store trained pipelines.
        self.Pipelines = None
        self.second_level_models={}
        #List of selected labels for classification.
        self.selected_Labels = constants.selected_labels
        #Flag to indicate if text data should be preloaded from train\test or recalculated.
        self.PreLoadText=False
        #Internal flag for text preloading.
        self.set_preload_text_= False
        #indicating the succesfull loading of the classifier

        #env types data
        self.env_types =[]
        print("init classifier")

    #Sets the preload text flag    
    def set_preload_text(self,value=True):
        """
        Sets the set_preload_text_ attribute to determine whether to preload text
        Args:
            value (bool): Flag to set the preload text. Default is True.
        """
        self.set_preload_text_=value

    #Labels elements not in the provided selected_labels list as 'Others'.
    def label_other(self,x,labels =constants.selected_labels,label_mapping={}):
        """Labels the input x as 'Others' if it is not in the provided list of labels.

        Args:
            o	x (str): Input label.
            o	labels (list of str): List of selected labels.

        Returns:
            _type_: str
        """
        if str(x) in labels:
            return x
        else:
            return 'Others'
            
    #Loads data for training from separate files and calculating the last classification classes column.
    def Load_data(self):
        """
            Loads and preprocesses the data from the file system.
        """
        #get files from 2 separate sources and compining them
        self.data = get_all_files()
        print(self.data['PROJECT'])
        print("Data prepared")
        previously_extracted_text = os.listdir(constants.result_text_path)
        previously_extracted_text = [str(i.split(sep='.')[0]) for i in previously_extracted_text]
        
        #handelling PDF_ID column datatype
        self.data['PDF_ID'] = self.data['PDF_ID'].astype(str)
        self.data['PDF_ID']=self.data['PDF_ID'].apply(lambda x: str(x).replace('.0',''))
        print(self.data.head()['PDF_ID'])

        #use only Previously extracted pdfs 
        mask = self.data['PDF_ID'].isin(previously_extracted_text)
        self.data = self.data[mask]

        # Apply a function to label elements not in the list as 'other'
        self.data['classification'] = self.data.apply(lambda x: self.label_other(x['PROJECT']), axis=1)
        print(self.data['classification'])
        print("Data Loaded")


    #Loads text data to dataframe from text files
    def load_Text(self):
        """
        Loads the text data for each PDF ID.
        """
        self.data['text'] = self.data['PDF_ID'].astype(str).apply(lambda x: read_text(constants.result_text_path +str(x) + '.txt'))
        
        self.data = self.data[self.data['text']!=""]
    
    #check the counts of all classes in the classification types
    def analyze(self):
        """
        Analyzes the data by printing information and value counts.
        """
        self.data[self.data['datasheet']==1]['PROJECT'] ='Datasheet' 
        info = self.data.info()
        valus_count = self.data['PROJECT'].value_counts()
        valus_count.to_csv("Log\\total_data"+str(current_hour.hour)+".csv")
        info.to_csv("Log\\total_data_info"+str(current_hour.hour)+".csv")

        pass
    def train_multi_level_environmental(self,Classes_sample=['Datasheet','Package Brief','Manual and Guide','Application Brief','Product Brief','Package Drawing'],notSelected_labels =['Others'],class_weights={},grouping=1,Model_name="Environmental",sampling=1,load_trained=1):
        """
        Trains multi-level classifiers on the provided class samples.

        Args:
            o	Classes_sample (list of str): List of class samples. Defaults to ['Datasheet','Package Brief','Manual and Guide'].
            o	selected_classes (list of str): List of selected classes. Defaults to ['Datasheet','Package Brief','Manual and Guide'].
            o	Model_name (str): Model name. Defaults to "Datasheet".
        """

        for c in class_weights:
            constants.class_weights[c]=class_weights[c]
        if True or not (os.path.exists("data\\train\\train.csv") or os.path.exists("data\\train\\train.csv")) :
            #if train and test dpesn't exist split the original data to train and test
            if self.set_preload_text_:
                print("gotData...")
                if os.path.exists("data\\debug_data_out.csv"):
                    self.data = pd.read_csv("data\\debug_data_out.csv",encoding="utf-8",low_memory=False)
                    self.data = self.data[self.data['classification']=='Environmental']
                    print(len(constants.environmental_types))
                    self.data['Others']=False
                    self.env_types.append('Others')
                    for index in  range(len(constants.environmental_types)):
                        col = constants.environmental_types[index]
                        type_list = self.data['PROJECT'].str.contains(col, case=False)
                        if sum(type_list)>10:
                            self.data[col]=type_list
                            self.env_types.append(col)
                        else:
                            print("Others label used ",col)
                            self.data['Others'] = self.data['Others']|type_list
                            

                    print(self.data.columns)
                else:
                    pass
                   
            else:
                print("loading Text....")
                self.load_Text()
                
            #remove rows woth no text
            traindata = self.data.dropna(subset=['text'])
            traindata = traindata[traindata['text']!='']
            traindata = traindata[traindata['text'].apply(len)>constants.min_Letters_in_text]
            traindata = traindata.drop_duplicates(subset=['PDF_URL'])
            X = traindata
            print(self.env_types)
            y = traindata[self.env_types]
            
            #splitting the training and test data with balance selection
            X_train_f, X_test_f, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)
            
            #save the training and testeing data
            X_train_f.to_csv("data\\train\\train.csv",index=False)
            X_test_f.to_csv("data\\test\\test.csv",index=False) 
            
        else:
            #is the paths exist just load the train and test data
            X_train_f = pd.read_csv("data\\train\\train.csv",low_memory=False)
            y_train = X_train_f[self.env_types]
            
            X_test_f = pd.read_csv("data\\test\\test.csv",low_memory=False)
            
            y_train = X_train_f[self.env_types]
            y_test = X_test_f[self.env_types]
            
        # Balance the training data using RandomUnderSampler
        X_train = X_train_f['text']
        X_test = X_test_f['text']
        
        
        # Initialize the resampled training data to the original training data
        X_train_resampled = X_train
        y_train_resampled =   y_train
        for col in self.env_types:
            print(y_train[col].value_counts())
        
        print("=============================================")
        print("=============================================")
        print(type(y_train))
        print(y_train_resampled)
        
        y_train:pd.DataFrame
        y_train_combine = y_train.astype(str).agg(','.join, axis=1)
        # Perform sampling if specified
        if sampling==1:
            print("oversampling..")
            sampler = RandomOverSampler(random_state=42,sampling_strategy='not majority')
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train.values.reshape(-1, 1), y_train_combine)
            y_train_resampled = [str(i).split(',') for i in y_train_resampled]
            y_train_resampled = pd.DataFrame(y_train_resampled,columns=self.env_types).astype(bool)
            X_train_resampled = pd.DataFrame(X_train_resampled,columns=['Text'])['Text'].astype(str)
        elif sampling==-1:
            print("undersampling...")
            sampler = RandomUnderSampler(random_state=42)
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train.values.reshape(-1, 1), y_train_combine)
            y_train_resampled = X_train_resampled[self.env_types]
            X_train_resampled = pd.DataFrame(X_train_resampled,columns=['Text'])['Text'].astype(str)
        print(y_train_resampled.value_counts())
        print('training the model')
        print("training "+"_model")
        # Train and evaluate each classifier
        for name, clf in self.env_classifiers:
            print("training "+name+"_model")
            # Create a pipeline for each classifier
            clf_pipeline = Pipeline([
                ('preprocessor',TextPreprocessor()),
                ('tfidf', self.train_tfidf_vectorizer),
                ('select_best', SelectKBest(chi2, k=200)),
                ('clf', clf)
            ])
            
            print(clf_pipeline)

            # Special handling for XGBoost classifier
            if name=='XGB':
                # encoding data
                encoded_data = self.label_encoder.transform(y_train_resampled)

                y_train_resampled = encoded_data
            # Train the model
            print(len(X_train_resampled),len(y_train_resampled))
            if load_trained==0:
                print(type(X_train_resampled),type(y_train_resampled))
                print(y_train_resampled.info())
                clf_pipeline.fit(X_train_resampled, y_train_resampled)
            elif load_trained==1:
                self.load_models(date="")
                clf_pipeline = self.Pipelines[name][1]
                print(clf_pipeline)
            
            
            
            
            print("-----------------------")
            

            #-----------------------------------------------
            
            print("--------------------------------")

            # Evaluate the model
            y_pred = clf_pipeline.predict(X_test)
            # Get probabilities and apply threshold
            
            print("--------------------------------")
            
            print(y_test)
            print(y_pred)
            # Transform predictions for XGBoost
            if name=='XGB':
                # Transform the test labels
                try:
                    y_pred = self.label_encoder.inverse_transform(y_pred)
                    y_train_pred = self.label_encoder.inverse_transform(y_train_pred)
                except ValueError as e:
                    unseen_labels = set(y_pred) - set(self.label_encoder.classes_)
                    print(f"Unseen labels: {unseen_labels}")
             

            # Print classification report for test data
            print(f"Classification Report for {name}:")
            print(len(X_test))
            for index,env_class in enumerate(self.env_types):
                print("Classification Report for ",env_class)
                print(classification_report(y_test[env_class], y_pred[:,index],zero_division=0))

            """y_train_pred = clf_pipeline.predict(X_train_resampled)
            #-----------------------------------------------
            #y_train_pred_proba = clf_pipeline.predict_proba(X_train_resampled)[:, 1]
            #y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
            #-----------------------------------------------

            
            # Print classification report for training data
            print("classification Train Report:")
            try:
                if  name=='XGB':
                    y_train_resampled = self.label_encoder.inverse_transform(y_train_resampled)
                print(classification_report(y_train_resampled, y_train_pred))
            except Exception as e:
                print(e)
            """
            #prediction result saving
            
            if load_trained!=1:
                # Save the trained model
                model_filename = f"models\\{Model_name}_{name}_model"+str(current_hour.day)+'_'+str(current_hour.month)+'_'+str(current_hour.hour)+".pkl"
                joblib.dump(clf_pipeline, model_filename)
                model_filename = f"models\\{Model_name}_{name}_model.pkl"
                joblib.dump(clf_pipeline, model_filename)
                
                print(f"Model for {name} saved as {model_filename}")


    #Trains multiple classifiers on hierarchical classes.
    def train_multi_level(self,Classes_sample=['Life_Cycle','News','Application Brief','Product Brief','Package Drawing','PCN','Package Brief','Others','Datasheet','Environmental'],notSelected_labels =['Datasheet','Environmental'],class_weights={},grouping=1,Model_name="Others",sampling=1,load_trained=1):
        """
        Trains multi-level classifiers on the provided class samples.

        Args:
            o	Classes_sample (list of str): List of class samples. Defaults to ['Datasheet','Package Brief','Manual and Guide'].
            o	selected_classes (list of str): List of selected classes. Defaults to ['Datasheet','Package Brief','Manual and Guide'].
            o	Model_name (str): Model name. Defaults to "Datasheet".
        """
        self.env_classifiers = [
            #('XGB',xgb.XGBClassifier(verbosity=1,n_estimators=constant.n_estimators,max_depth=30,subsample=0.6,eval_metric='mlogloss',objective='multi:softmax')),
            ('RandomForest', RandomForestClassifier(random_state=42,n_estimators=constants.n_estimators,max_depth=40,min_samples_split=15,min_samples_leaf=5,max_features='sqrt',class_weight=class_weights)),
        ]
        for c in class_weights:
            constants.class_weights[c]=class_weights[c]
        if not (os.path.exists("data\\train\\train.csv") or os.path.exists("data\\train\\train.csv")):
            #if train and test dpesn't exist split the original data to train and test
            if self.set_preload_text_:
                print("gotData...")
                if os.path.exists("data\\debug_data_out.csv"):
                    self.data = pd.read_csv("data\\debug_data_out.csv",encoding="utf-8",low_memory=False)
                    self.data['classification'] = self.data.apply(lambda x: self.label_other(x['classification'],Classes_sample), axis=1) 
                    print(self.data['classification'].unique())
                else:
                    pass
                   
            else:
                print("loading Text....")
                self.load_Text()
                
            #remove not selected labels 
            mask=self.data['classification'].isin(notSelected_labels)
            self.data = self.data[~mask]
            print('1',self.data['classification'].unique())
            self.data=self.data
            self.data = self.data[self.data['PROJECT']!='In_Complete_DS']
            
            #remove rows woth no text
            traindata = self.data.dropna(subset=['text'])
            traindata = traindata[traindata['text']!='']
            print('1.5',traindata['classification'].unique())
            traindata = traindata[traindata['text'].apply(len)>constants.min_Letters_in_text]
            print('2',traindata['classification'].unique())
            #remove duplicates
            print('5',traindata['classification'].value_counts())
            traindata = traindata.drop_duplicates(subset=['PDF_URL'])
            print('6',traindata['classification'].value_counts())
            X = traindata
            y = traindata['classification']
            print(y.value_counts())

            #fit and save the encoder of the classifications
            self.label_encoder.fit_transform(y)
            joblib.dump(self.label_encoder,'models\\'+Model_name+'_Label_ecoders.pkl')
            
            #splitting the training and test data with balance selection
            X_train_f, X_test_f, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=42,stratify=y)
            
            #save the training and testeing data
            X_train_f.to_csv("data\\train\\train.csv",index=False)
            X_test_f.to_csv("data\\test\\test.csv",index=False) 
            X_train_f['classification'] = X_train_f.apply(lambda x: self.label_other(x['classification']), axis=1)
            X_test_f['classification'] = X_test_f.apply(lambda x: self.label_other(x['classification']), axis=1)
            
        else:
            #is the paths exist just load the train and test data
            X_train_f = pd.read_csv("data\\train\\train.csv",low_memory=False)
            X_train_f['classification'] = X_train_f.apply(lambda x: self.label_other(x['PROJECT_1'],Classes_sample), axis=1)
            y_train = X_train_f['classification']
            print(y_train.unique())
            print("train sample",len(X_train_f))
            X_test_f = pd.read_csv("data\\test\\test.csv",low_memory=False)
            X_test_f['classification'] = X_test_f.apply(lambda x: self.label_other(x['PROJECT_1'],Classes_sample), axis=1)
            
            y_train = X_train_f['classification']
            y_test = X_test_f['classification']
            print(y_test.unique())

            #fitting the label encoder to the data
            self.label_encoder.fit_transform(y_train)
            joblib.dump(self.label_encoder,'models\\'+Model_name+'_Label_ecoders.pkl')
        # Balance the training data using RandomUnderSampler
        X_train = X_train_f['text']
        X_test = X_test_f['text']
        
        
        # Initialize the resampled training data to the original training data
        X_train_resampled = X_train
        y_train_resampled =   y_train


        # Perform sampling if specified
        if sampling==1:
            print("oversampling..")
            sampler = RandomOverSampler(random_state=42,sampling_strategy='not majority')
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train.values.reshape(-1, 1), y_train)
            X_train_resampled = pd.DataFrame(X_train_resampled,columns=['Text'])['Text'].astype(str)
        elif sampling==-1:
            print("undersampling...")
            sampler = RandomUnderSampler(random_state=42)
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train.values.reshape(-1, 1), y_train)
            X_train_resampled = pd.DataFrame(X_train_resampled,columns=['Text'])['Text'].astype(str)
        print(y_train_resampled.value_counts())
        print('training the model')
        print("training "+"_model")
        # Train and evaluate each classifier
        for name, clf in self.env_classifiers:
            print("training "+name+"_model")
            # Create a pipeline for each classifier
            clf_pipeline = Pipeline([
                ('preprocessor',TextPreprocessor()),
                ('tfidf', self.train_tfidf_vectorizer),
                ('select_best', SelectKBest(chi2, k=200)),
                ('clf', clf)
            ])
            
            print(clf_pipeline)

            # Special handling for XGBoost classifier
            if name=='XGB':
                # encoding data
                encoded_data = self.label_encoder.transform(y_train_resampled)

                y_train_resampled = encoded_data
            # Train the model
            print(len(X_train_resampled),len(y_train_resampled))
            if load_trained==0:
                clf_pipeline.fit(X_train_resampled, y_train_resampled)
            elif load_trained==1:
                self.load_models(date="")
                clf_pipeline = self.Pipelines[name][1]
                print(clf_pipeline)
            
            
            
            
            print("-----------------------")
            

            #-----------------------------------------------
            
            print("--------------------------------")

            # Evaluate the model
            y_pred = clf_pipeline.predict(X_test)
            # Get probabilities and apply threshold
            y_pred_prob = clf_pipeline.predict_proba(X_test)
            
            print("--------------------------------")
            y_pred_prob = [True if max(i) > 0.8 else False for i in y_pred_prob]
            
            
            #-----------------------------------------------
            #y_pred_prob = [max(i) for i in clf.predict_proba(X_test)]
            y_pred = y_pred[y_pred_prob]
            y_test = y_test[y_pred_prob]
            #-----------------------------------------------
            X_test_f = X_test_f[y_pred_prob]

            
            # Transform predictions for XGBoost
            if name=='XGB':
                # Transform the test labels
                try:
                    y_pred = self.label_encoder.inverse_transform(y_pred)
                    y_train_pred = self.label_encoder.inverse_transform(y_train_pred)
                except ValueError as e:
                    unseen_labels = set(y_pred) - set(self.label_encoder.classes_)
                    print(f"Unseen labels: {unseen_labels}")
             

            # Print classification report for test data
            print(f"Classification Report for {name}:")
            print(len(X_test))
            print(classification_report(y_test, y_pred))

            """y_train_pred = clf_pipeline.predict(X_train_resampled)
            #-----------------------------------------------
            #y_train_pred_proba = clf_pipeline.predict_proba(X_train_resampled)[:, 1]
            #y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
            #-----------------------------------------------

            
            # Print classification report for training data
            print("classification Train Report:")
            try:
                if  name=='XGB':
                    y_train_resampled = self.label_encoder.inverse_transform(y_train_resampled)
                print(classification_report(y_train_resampled, y_train_pred))
            except Exception as e:
                print(e)
            """
            #prediction result saving
            X_test_f['model_prediction'] = y_pred
            X_test_f['text']=''
            # Save the DataFrame to a CSV file
            X_test_f.to_csv('Results\\'+Model_name+'_classification_results.csv', index=False)
            if load_trained!=1:
                # Save the trained model
                model_filename = f"models\\{Model_name}_{name}_model"+str(current_hour.day)+'_'+str(current_hour.month)+'_'+str(current_hour.hour)+".pkl"
                joblib.dump(clf_pipeline, model_filename)
                model_filename = f"models\\{Model_name}_{name}_model.pkl"
                joblib.dump(clf_pipeline, model_filename)
                
                print(f"Model for {name} saved as {model_filename}")


    
    #Loads training and test data
    def load_train_test_data(self):
        """Loads and splits the data into training and testing sets.

        Returns:
            tuple (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
        """
        #check if train and test data exist in paths
        if not (os.path.exists("data\\train\\train.csv") or os.path.exists("data\\train\\train.csv")):
            #if train and test dpesn't exist split the original data to train and test
            if self.set_preload_text_:
                print("gotData...")
                if os.path.exists("data\\debug_data_out.csv"):
                    self.data = pd.read_csv("data\\debug_data_out.csv",encoding="utf-8",low_memory=False)
                    self.data['classification'] = self.data.apply(lambda x: self.label_other(x['classification']), axis=1)
                    print(self.data['classification'].unique())
                else:
                    pass
                   
            else:
                print("loading Text....")
                self.load_Text()
                
            #remove not selected labels 
            mask=self.data['classification'].isin(constants.not_selected_Labels)
            self.data = self.data[~mask]
            print('1',self.data['classification'].unique())
            self.data=self.data
            self.data = self.data[self.data['PROJECT']!='In_Complete_DS']
            
            #remove rows woth no text
            traindata = self.data.dropna(subset=['text'])
            traindata = traindata[traindata['text']!='']
            print('1.5',traindata['classification'].unique())
            traindata = traindata[traindata['text'].apply(len)>constants.min_Letters_in_text]
            print('2',traindata['classification'].unique())
            #remove duplicates
            print('5',traindata['classification'].value_counts())
            traindata = traindata.drop_duplicates(subset=['PDF_URL'])
            print('6',traindata['classification'].value_counts())
            X = traindata
            y = traindata['classification']
            print(y.value_counts())

            #fit and save the encoder of the classifications
            self.label_encoder.fit_transform(y)
            joblib.dump(self.label_encoder,'models\\Label_ecoders.pkl')
            
            #splitting the training and test data with balance selection
            X_train_f, X_test_f, y_train, y_test = train_test_split(X, y, test_size=.99, random_state=42,stratify=y)
            
            #save the training and testeing data
            X_train_f.to_csv("data\\train\\train.csv",index=False)
            X_test_f.to_csv("data\\test\\test.csv",index=False) 
            X_train_f['classification'] = X_train_f.apply(lambda x: self.label_other(x['classification']), axis=1)
            X_test_f['classification'] = X_test_f.apply(lambda x: self.label_other(x['classification']), axis=1)
            
        else:
            #is the paths exist just load the train and test data
            X_train_f = pd.read_csv("data\\train\\train.csv",low_memory=False)
            X_train_f['classification'] = X_train_f.apply(lambda x: self.label_other(x['classification']), axis=1)
            y_train = X_train_f['classification']
            print(y_train.unique())
            print("train sample",len(X_train_f))
            X_test_f = pd.read_csv("data\\test\\test.csv",low_memory=False)
            X_test_f['classification'] = X_test_f.apply(lambda x: self.label_other(x['classification']), axis=1)
            
            y_train = X_train_f['classification']
            y_test = X_test_f['classification']
            print(y_test.unique())

            #fitting the label encoder to the data
            self.label_encoder.fit_transform(y_train)
            joblib.dump(self.label_encoder,'models\\Label_ecoders.pkl')
        return X_train_f,X_test_f,y_train, y_test
        

    #Trains the classifiers on the dataset
    def train(self,sampling=1,load_trained=1):
        """
            Trains and evaluates the classifiers on the training data.
        Args:
            sampling (int, optional): choosing sampling method 1:overSampling, -1:undersampling. Defaults to 1.
        """

        # Print a message indicating that training is starting
        print("training")

        # Load the training and testing data
        X_train_f, X_test_f, y_train, y_test =self.load_train_test_data()
        X_train = X_train_f['text']
        X_test = X_test_f['text']
        
        
        # Initialize the resampled training data to the original training data
        X_train_resampled = X_train
        y_train_resampled =   y_train


        # Perform sampling if specified
        if sampling==1:
            print("oversampling..")
            sampler = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train.values.reshape(-1, 1), y_train)
            X_train_resampled = pd.DataFrame(X_train_resampled,columns=['Text'])['Text'].astype(str)
        elif sampling==-1:
            print("undersampling...")
            sampler = RandomUnderSampler(random_state=42)
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train.values.reshape(-1, 1), y_train)
            X_train_resampled = pd.DataFrame(X_train_resampled,columns=['Text'])['Text'].astype(str)

            
        # Print the length and content of the training data
        print(len(X_train),X_train)
        
        print('training the model')
        print("training "+"_model")
        # Train and evaluate each classifier
        for name, clf in self.classifiers:
            print("training "+name+"_model")
            # Create a pipeline for each classifier
            clf_pipeline = Pipeline([
                ('preprocessor', TextPreprocessor()),
                ('tfidf', self.train_tfidf_vectorizer),
                ('select_best', SelectKBest(chi2, k=constants.selected_features)),
                ('clf', clf)
            ])
            print(clf_pipeline)

            # Special handling for XGBoost classifier
            if name=='XGB':
                # encoding data
                encoded_data = self.label_encoder.transform(y_train_resampled)

                y_train_resampled = encoded_data
            # Train the model
            print(len(X_train_resampled),len(y_train_resampled))
            if load_trained==0:
                clf_pipeline.fit(X_train_resampled, y_train_resampled)
            elif load_trained==1:
                self.load_models(date="")
                clf_pipeline = self.Pipelines[name][1]
                print(clf_pipeline)
            
            
            
            
            print("-----------------------")
            

            #-----------------------------------------------
            
            print("--------------------------------")

            # Evaluate the model
            y_pred = clf_pipeline.predict(X_test)
            # Get probabilities and apply threshold
            y_pred_prob = clf_pipeline.predict_proba(X_test)
            
            print("--------------------------------")
            y_pred_prob = [True if max(i) > 0.8 else False for i in y_pred_prob]
            
            
            #-----------------------------------------------
            #y_pred_prob = [max(i) for i in clf.predict_proba(X_test)]
            y_pred = y_pred[y_pred_prob]
            y_test = y_test[y_pred_prob]
            #-----------------------------------------------
            X_test_f = X_test_f[y_pred_prob]

            
            # Transform predictions for XGBoost
            if name=='XGB':
                # Transform the test labels
                try:
                    y_pred = self.label_encoder.inverse_transform(y_pred)
                    y_train_pred = self.label_encoder.inverse_transform(y_train_pred)
                except ValueError as e:
                    unseen_labels = set(y_pred) - set(self.label_encoder.classes_)
                    print(f"Unseen labels: {unseen_labels}")
             

            # Print classification report for test data
            print(f"Classification Report for {name}:")
            print(len(X_test))
            print(classification_report(y_test, y_pred))

            """y_train_pred = clf_pipeline.predict(X_train_resampled)
            #-----------------------------------------------
            #y_train_pred_proba = clf_pipeline.predict_proba(X_train_resampled)[:, 1]
            #y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
            #-----------------------------------------------

            
            # Print classification report for training data
            print("classification Train Report:")
            try:
                if  name=='XGB':
                    y_train_resampled = self.label_encoder.inverse_transform(y_train_resampled)
                print(classification_report(y_train_resampled, y_train_pred))
            except Exception as e:
                print(e)
            """
            #prediction result saving
            X_test_f['model_prediction'] = y_pred
            X_test_f['text']=''
            # Save the DataFrame to a CSV file
            X_test_f.to_csv('Results\\classification_results.csv', index=False)
            if load_trained!=1:
                # Save the trained model
                model_filename = f"models\\{name}_model"+str(current_hour.day)+'_'+str(current_hour.month)+'_'+str(current_hour.hour)+".pkl"
                joblib.dump(clf_pipeline, model_filename)
                model_filename = f"models\\{name}_model.pkl"
                joblib.dump(clf_pipeline, model_filename)
                
                print(f"Model for {name} saved as {model_filename}")


    def apply_thresholds(self, probabilities, thresholds):
        """
        Apply thresholds to the predicted probabilities for multi-class classification.
        Args:
            probabilities (ndarray): Predicted probabilities for each class.
            thresholds (dict): A dictionary specifying classification thresholds for each class.

        Returns:
            ndarray: Predicted classes based on the applied thresholds.
        """
        if thresholds is None:
            return probabilities.argmax(axis=1)
        
        num_samples = probabilities.shape[0]
        num_classes = probabilities.shape[1]
        predictions = np.zeros(num_samples, dtype=int)
        
        for i in range(num_samples):
            probs = probabilities[i]
            for cls, threshold in thresholds.items():
                if probs[self.label_encoder.transform([cls])[0]] >= threshold:
                    predictions[i] = self.label_encoder.transform([cls])[0]
                    break
        
        return predictions
    #Loads pre-trained models.
    def load_models(self,second_level=['Datasheet','Environmental','Others'],prefix="",date="7_6_12"):
        """
        Loads the trained models from the file system.

        Args:
            o	date (str): Date string used to load specific model versions. Defaults to "7_6_12".

        Returns:
            _type_: int
        """
        print(len(self.classifiers))
        #loading label encoder
        self.label_encoder=joblib.load(f'models\\{prefix}Label_ecoders.pkl')
        self.Pipelines={}
        for (name, clf) in self.classifiers:
            try:
                #loading model
                print(f'{name}_model'+date+'.pkl')
                clf = joblib.load(f'models\\{prefix}{name}_model'+date+'.pkl')
                self.Pipelines[name]=(name,clf)
                self.print_model(name,clf)
            except Exception as e :
                print("exception: "+str(e),":",traceback.format_exc())
                return 0
        for labels in second_level:
            clf = joblib.load(f'models\\{labels}_{name}_model'+date+'.pkl')
            self.second_level_models[labels]=clf
        return 1
    

    #Prints model parameters and features.
    def print_model(self,name,model):
        """Prints the parameters and selected features of the model

        Args:
            o	name (str): Name of the model.
            o	model (Pipeline): Model pipeline.
        """
        if name =='XGB':
            #printing xgb parameters 
            print(model['clf'].get_xgb_params())
            #print tfidf features
            x = model['tfidf'].get_feature_names_out()
            print(list(x))
            #print best features selected from td-idf
            print(x[model['select_best'].get_support()])
            pass
        else:
            print(model['clf'].get_params())
            tree = model['clf'].estimators_[0].tree_
            attributes = [attr for attr in dir(tree) if not attr.startswith('_')]
            
            x = model['tfidf'].get_feature_names_out()
            print(list(x))
            print(x[model['select_best'].get_support()])
            pass
        pass


    #Preprocesses input text.
    def preprocess(self, text):
        """
        Transforms the input text using the TF-IDF vectorizer

        Args:
            o	text (str): Input text.

        Returns:
            _type_: sparse matrix
        """
        return self.train_tfidf_vectorizer.transform([text])
    

    #Predicts the class of a PDF document from its URL.
    def predict_pdf(self,url):
        """
        Extracts text from the PDF at the given URL and makes predictions.

        Args:
            o	url (str): URL of the PDF.

        Returns:
            _type_: list of str
        """
        pdf_link = url
        if is_valid_pdf_link(pdf_link):
            num_of_pages,extracted_text = extract_text_from_pdf_Production(pdf_link)
            if extracted_text is not None and extracted_text!="":
                print("text extracted successfully")
                print(extracted_text)
                
                predictions = self.predict([extracted_text])

                print(predictions)
                return 1, predictions
            else: 
                print("data can't be extracted, file must be searchable.")
                return 0,"data can't be extracted, file must be searchable."
        else:
            print("invalid url, url must contain siliconecpert domain and be pdf file")
            return 0,"invalid url, url must contain siliconecpert domain and be pdf file"
        

    #Predicts the class of multiple PDF documents from a DataFrame of URLs.
    def predict_Files(self,df:pd.DataFrame):
        """
        Extracts text from the PDFs, makes predictions, and returns the results.

        Args:
            o	df (pd.DataFrame): DataFrame containing URLs of PDFs

        Returns:
            _type_: pd.DataFrame
        """
        predictions = []
        extracted_df =[]
        num_pages=[]
        extracted_len=[]
        urls=[]
        df['Extracted']=0
        for index,row in df.iterrows():
            url = row['URL']
            pdf_link = url
            
            if is_valid_pdf_link(pdf_link):
                print("Valid Link")
                num_of_pages,extracted_text = extract_text_from_pdf_Production(pdf_link)
            else:
                num_of_pages=0
                extracted_text=""
            num_pages.append(num_of_pages)
            extracted_df.append(extracted_text)
            extracted_len.append(len(extracted_text))
            urls.append(url)
            if (extracted_text is not None) and len(extracted_text)>constants.min_Letters_in_text:
                print("text extracted successfully", index)
                
                
                # Assign the new value to the new column
                df.at[index, 'Extracted'] = 1

                
            else:
                print("invalid url, url must contain siliconecpert domain and be pdf file")
        
        df.insert(0,'Index',[ i for i in range(len(df)) ])
        original_df = df
        original_df['text']=extracted_df
        predictions= self.predict(extracted_df)
        predictions = pd.DataFrame(predictions)


        for column in predictions.columns:
            original_df.insert(2, column, predictions[column].to_list(), True)
        original_df.to_csv(r'C:\Users\161070\Downloads\datatext.csv')
        original_df['textLenght'] = [len(i) for i in original_df['text'].replace(pd.NA,"")]
        original_df['numOfPages'] = '-'
        print("calculating length")
        print(original_df['textLenght'])
        len_reliability = original_df['textLenght'].to_list()>constants.min_Letters_in_text
        original_df['Searchable'] = len_reliability
        """print(original_df['model_confidence'])
        list_relibility = original_df['model_confidence'].to_list() > 0.8
        list_relibility = [i and j for i,j in zip(list_relibility,len_reliability)]
        print("list reliablilty")
        print(list_relibility)"""
        original_df['reliable'] = (original_df['model_confidence']> 0.8) & original_df['Searchable']
        print("list reliablilty done")
        
        # Function to check if language is English
        def is_english(text):
            try:
                language, score = langid.classify(text)
                return language
            except:
                return 'cant detect'


        conditions = [
        (original_df['textLenght'] < constants.min_Letters_in_text),
        (original_df['textLenght'] > constants.min_Letters_in_text) & (original_df['reliable'] == False),
        (original_df['textLenght'] > constants.min_Letters_in_text) & (original_df['reliable'] == True)
        ]
        
        # Define the corresponding values for each condition
        values = [
            'Not searchable',
            'Low confidence Level',
            'High confidence Level'
        ]
        print("checking language")
        original_df['lang']=    original_df['text'].apply(is_english)
        # Create the new column 'Trust_Level' using np.select
        original_df['Trust_Level flag'] = np.select(conditions, values)
        
        original_df= original_df.drop(columns=['text'])
        
        return original_df
        
    def predict_text_df(self,df:pd.DataFrame):
        original_df=df
        print("predicting...")
        text_list = original_df['text'].replace(pd.NA,"None").to_list()
        
        predictions= self.predict(text_list)
        
        print("prediction done.")
        
        predictions = pd.DataFrame(predictions)
        
        for column in predictions.columns:
            original_df.insert(2, column, predictions[column].to_list(), True)

        original_df['textLenght'] = [len(i) for i in original_df['text'].replace(pd.NA,"")]
        original_df['numOfPages'] = '-'
        
        len_reliability = original_df['textLenght']>constants.min_Letters_in_text
        print(len_reliability)
        original_df['Searchable'] = len_reliability
        print(original_df['model_confidence'])
        #list_relibility = original_df['model_confidence']> 0.8 & original_df['Searchable']
        original_df['reliable'] = (original_df['model_confidence']> 0.8) & original_df['Searchable']
        
        print(original_df['reliable'])
        # Function to check if language is English
        def is_english(text):
            try:
                lang, conf = langid.classify(text)
                return  lang
            except:
                return 'cant detect'
        
        conditions = [
        (original_df['textLenght'] < constants.min_Letters_in_text),
        (original_df['textLenght'] > constants.min_Letters_in_text) & (original_df['reliable'] == False),
        (original_df['textLenght'] > constants.min_Letters_in_text) & (original_df['reliable'] == True)
        ]
        
        # Define the corresponding values for each condition
        values = [
            'Not searchable',
            'Low confidence Level',
            'High confidence Level'
        ]
        print("extracting_english")
        
        #original_df['lang']=    original_df['text'].apply(is_english)
        # Create the new column 'Trust_Level' using np.select
        original_df['Trust_Level flag'] = np.select(conditions, values)
        dfs = self.second_level_predict(original_df)
        for name in dfs:
            df = dfs[name]
            df= df.drop(columns=['text'])
            df.to_csv(f"Results/{name}.csv")
        
        original_df= original_df.drop(columns=['text'])
        
        
        return original_df
    

    def second_level_predict(self,original_df:pd.DataFrame):
        result= {}
        for model_name in self.second_level_models:
            data_ = original_df.copy()
            print("predicting",model_name)
            data = data_[data_['model_Prediction']==model_name]
            text_list = data['text'].replace(pd.NA,'').to_list()
            model = self.second_level_models[model_name]
            try:
                print(model)
                df = model.predict(text_list)
                if model_name=='Environmental':
                    print(df)
                    df=pd.DataFrame(df,columns=constants.environmental_types)
                    print(df)
                    data = pd.concat([data,df],axis=1)
                else:
                    print(len(df))
                    data['2nd Prediction'] = df
                
                result[model_name]=data
            except Exception as e:
                print(traceback.format_exc())
        return result
        pass
    #Predicts the class of input text.
    def predict(self, text_list=[]):
        """Makes predictions using the loaded models for each input text

        Args:
            o	text_list (list of str): List of input texts. Defaults to [].

        Returns:
            _type_: dict
        """
        # Make predictions
        predictions = {}
        confidence_scores={}
        
        for name in self.Pipelines:
            print("  model RF")
            clf = self.Pipelines[name][1]
            print("    predicting...")
            predictions['model_Prediction'] = clf.predict(text_list)
            print("    prediction done.")
            if name =="XGB":
                predictions["model_Prediction"] = self.label_encoder.inverse_transform(predictions[name])
            predictions["model_confidence"] = [max(i) for i in clf.predict_proba(text_list)]
            print("returning")
        return predictions 
    
    #Expands the models by loading pre-trained ones and printing their parameters.
    def expand_Models(self):
        """Loads the models and returns the RandomForestClassifier with best parameters from GridSearchCV

        Returns:
            _type_: RandomForestClassifier
        """
        self.load_models()
        name ,clf = self.Pipelines['RandomForest_search']
        clf:Pipeline
        grid_search = clf.named_steps['clf']

        
        # Use GridSearchCV to find the best parameters
        rf_model:RandomForestClassifier = grid_search.best_estimator_
        print(rf_model.get_params())
        return rf_model
    

