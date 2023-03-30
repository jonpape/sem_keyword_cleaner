import streamlit as st
import pandas as pd
import numpy as np
#import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import re


# Define a function to read a CSV file
def read_csv(file):
    df = pd.read_csv(file)
    return df

def keyword_cleaner(df):
    # Step - a : Remove blank rows if any.
    df['Keywords'].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    df['Keywords'] = [entry.lower() for entry in df['Keywords']]
    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    nltk.download('punkt')
    df['Keywords']= [word_tokenize(entry) for entry in df['Keywords']]
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(df['Keywords']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        df.loc[index,'Keywords_final'] = str(Final_words)
    
    # Step - e : Get the token counts
    df['Tokens'] = df['Keywords_final'].apply(lambda x: len(x.split(',')))
    # concatenate all the keywords into a single string
    for index,entry in enumerate(df['Keywords_final']):
        #entry = re.sub('[,]',' ',entry)
        entry = re.sub('[,\[\]\']','',entry)
        df.loc[index,'Keywords_final_str'] = entry
    df.drop_duplicates(subset='Keywords_final_str', keep='first', inplace=True)
    df = df.dropna()
    df_Keywords = df['Keywords_final_str', 'Tokens']
    df_Keywords = df_Keywords.rename(columns={"Keywords_final_str": "Keywords"})
    return df_Keywords

# Set the page title
st.set_page_config(page_title='SEM Keyword Cleaner', page_icon=':wrench:')
st.title('SEM Keyword Cleaner')
st.markdown('Upload a list of keywords and this app will clean them up for you. ' + \
            'App will remove duplicates, remove stop words, lemmatize the words, ' + \
            'and count the number of tokens.')

# Add a file uploader widget
file = st.file_uploader('Upload a CSV file with column name "Keywords"', type='csv')

# If a file is uploaded, display its contents
if file is not None:
    df_upload = read_csv(file)
    if 'Keywords' in df_upload.columns:
        # If the column exists, continue with your code
        df = df_upload[['Keywords']]
        df = keyword_cleaner(df)
        st.dataframe(df, width=1000, height=800)
    else:
        # If the column doesn't exist, raise an error message
        raise ValueError("The DataFrame doesn't have a 'Keywords' column")
    