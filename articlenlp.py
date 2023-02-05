import spacy
import nltk
import statistics as s
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from urllib.request import urlopen
from bs4 import BeautifulSoup
import streamlit as st

model = SentenceTransformer('all-MiniLM-L6-v2')
classifier = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model_token = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
token = pipeline("ner", model=model_token, tokenizer=tokenizer)



#Getting everything together
def get_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def user_summarize(article):
    summary = list(summarizer(article)[0].values())[0]
    return summary

def main():
    st.write("Filtering out negativity.\n")
    user_url = txt
    st.write("\nRetrieving information...please wait\n")
    article = get_article(str(user_url))
    summary = user_summarize(article)
    result = classifier(summary)
    st.write("Here's what we found: \n")
    st.write("The article can be summarised as follows: \n")
    st.write(summary)
    

    st.write("\nThe article is mostly: ", {list(result[0].values())[0]}, "with an accuracy of: ", {round(list(result[0].values())[1] * 100,4)}," %")

st.title('NewScan - analysing news like never before')

txt = st.text_area('Enter your news article link: ', '')

if st.button('Analyze'):
    main()
else:
    st.write('...')