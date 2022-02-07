import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
from tensorflow import keras





class Preprocessing:
    '''Preprocesses the given column and returns data that is ready to feed to the model'''
    
    def __init__(self):
        # initializing objects for different preprocessing techniques
        self.stop_words = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer_abstract = None
        self.vectorizer_title = None
        self.training = None
        
    

    def remove_punctuations(self, text):
        # removes unnecessary punctuations from the text
        text = text.lower()
        cleaned_text = re.findall("[a-zA-Z]+", text)
        
        return cleaned_text
    

    def stop_words_remover(self, text):
        # removes stopwords since they are not useful for sentiment prediction
        cleaned_text = [w for w in text if not w in self.stop_words]
        
        return cleaned_text
    
    
    def lemmatize(self, text):
        # brings words to their root words 
        cleaned_text = ' '.join([self.lemmatizer.lemmatize(i) for i in text])
        
        return cleaned_text
    
    
    def vectorize(self, X_cleaned): 
        # converting text to vectorized form
        if self.training=="title":
            X_vectorized = self.vectorizer_title.transform(X_cleaned)
        else:
            X_vectorized = self.vectorizer_abstract.transform(X_cleaned)
                
        return X_vectorized



    
    def preprocess(self, X, train_labels=None, training=None):
        # takes input column and applies different pre-processing techniques
        X_cleaned = pd.DataFrame()
        self.training = training
        X = X.apply(lambda x: self.remove_punctuations(x))
        X = X.apply(lambda x: self.stop_words_remover(x))
        X_cleaned = X.apply(lambda x: self.lemmatize(x))
        X_vectorized=self.vectorize(X_cleaned)
        
        return X_cleaned, X_vectorized
   


