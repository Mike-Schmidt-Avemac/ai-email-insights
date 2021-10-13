'''
MIT License

Copyright (c) 2021 Avemac Systems LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

########################################################################################
# This source contains content preparation and feature extraction/topic modeling routines
# for unsupervised data processing to generate a supervised classification output for
# subsequent model training.
#
# -- This is the supporting code for my email insights and analysis blog series available
# on my website at https://www.avemacconsulting.com/resources/.  Part 3 of the series at
# https://www.avemacconsulting.com/2021/09/23/email-insights-from-data-science-part-3/
#
# The code has been tested for topic modeling sentiment and professional alignment. Some
# feature extraction was also included, but not used during the model process.
#
# Three different sentiment methods are included.  Two of the methods are based upon
# sentiment vocabularies and cosine similarity.  The remaining method is rule-based
# and utilizes the NLTK Vader algorithm. Classes include "negative", "positive", 
# "neutral" and "unknown".
# 
# Regarding professional alignment classification, three methods were also implemented.
# Two are based upon Non-Negative Matrix Factorization and the third utilizes
# Latent Dirichlet Allocation plus cosine similarity.  Class labels include "fun", "work"
# and "unknown".
#
# -- The dataset used for this exercise was specifically created from the raw Enron
# email repository located at https://www.cs.cmu.edu/~enron/ with 3 labels generated
# for sentiment (positive/negative/neutral/unknown) and alignment(business/personal).  
# 
# The code for formatting the raw email content, performing basic analysis and creating
# the supervised dataset can be found in this Github repo with details referenced on my website.
# 
# Part 1. https://www.avemacconsulting.com/2021/08/24/email-insights-from-data-science-techniques-part-1/
# Part 2. https://www.avemacconsulting.com/2021/08/27/email-insights-from-data-science-part-2/
# Part 3. https://www.avemacconsulting.com/2021/09/23/email-insights-from-data-science-part-3/
# Part 4. https://www.avemacconsulting.com/2021/10/12/email-insights-from-data-science-part-4/
#
# ---- Classes ----
#  class UnsupervisedModeling - Includes all methods.
#
# ---- Methods ----
#  def _address_clean - Regex to filter invalid email addresses
#  def _fix_email_addresses - Control function to correct invalid email addresses
#  def _create_sentiment_dictionary - Routine to load one of the dictionaries used to classify email content for sentiment
#  def cliporpad - simple text formatting function
#
#  def _classify_sentiment - Logic that determines the polarity of an email based upon sentiment word associations.
#  def sentiment_analysis - Calculates topic relevant using CountVectorization and Latent Dirichlet Allocation.
#
#  def sentiment_analysis_vader - Rule-based sentiment analyzer.
#
#  def _classify_topic_method - Logic that determines the alignment of an email based upon topic weights and cosine similarity.
#  def _stop_word_support_function - Calculates the list of words not to include in analysis.
#  def _add_label_to_dataframe - Support function for building the output dataframe
#  def _body_content_analysis_tokenizer - Lemmatizer function for CountVectorizer
#  def topic_classification - Calculates alignment topic distribution using CountVectorizer and NMF/LDA.
#
# ---- Main ----
#  Processes for all sentiments and alignment classifications.
#
########################################################################################

#!/usr/bin/python3 -W ignore::DeprecationWarning
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from time import time
import datetime as dt
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, names, words
from nltk.sentiment import vader
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from utils.auto_classify import AutoClassify

pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_colwidth', 100)

KAGGLE_BASEDIR = '/kaggle'
LOCAL_BASEDIR = '/proto/learning/avemac/email_analysis_blog'
IS_LOCAL = not os.path.exists(KAGGLE_BASEDIR) # running in Kaggle Environment

#####################################################################
# Unsupervised Modeling Functions
#####################################################################

class UnsupervisedModeling():
    ''' Class for topic modeling unstructured data into labeled training data '''

    def __init__(self, config):
        self.data_dir = config['data_dir']
        self.word_filter_list_fn = config['word_filter_list_fn']
        self.manual_word_counts_fn = config['manual_word_counts_fn']
        self.topic_gradients_fn = config['topic_gradients_fn']
        self.plot_save_dir = config['plot_image_save_directory']
        self.sentiment_fn = config['sentiment_fn']

        raw_emails = pd.read_csv(self.data_dir + config['email_extracted_fn'])
        raw_emails.fillna(value="[]", inplace=True)

        self.email_df = self._fix_email_addresses('From_Address', raw_emails)
        self.email_df = self._fix_email_addresses('To_Address', raw_emails)
        self.email_df = self._fix_email_addresses('Cc_Address', raw_emails)
        self.email_df.pop('Bcc_Address') # from analysis phase, Bcc is duplicated - remove
        self.email_df.pop('Unnamed: 0') # artifact from analysis phase - remove

        self.email_df['DateTime'] = self.email_df['DateTime'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z'))
        self.email_df['DateTime_TS'] = self.email_df['DateTime'].apply(lambda x: x.timestamp())
        self.email_df['DateTime_HOUR'] = self.email_df['DateTime'].apply(lambda x: x.hour)
        self.email_df['DateTime_MONTH'] = self.email_df['DateTime'].apply(lambda x: x.month)

        # from the analysis phase - remove samples with content length greater than 6000 characters
        self.email_df = self.email_df[self.email_df['Body'].map(len) <= 6000].reset_index(drop=True)

        # build positive/negative sentiment dictionaries
        self.sentiment_d = self._create_sentiment_dictionary(self.data_dir, config['negative_sentiment_fn'], config['positive_sentiment_fn'])

        # common lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # auto classifier
        self.ac = AutoClassify(config=config)

        print(f'\n--- Init Complete\n\n')
        return

    def _address_clean(self, addr):
        ''' Additional email address cleaning '''
        addr = re.sub(r'e-mail <(.*)>',r'\1',addr)
        addr = re.sub(r' +', '', addr)
        addr = re.sub(r'/o.*=', '', addr)
        addr = re.sub(r'"', '', addr)
        return addr

    def _fix_email_addresses(self, type, df):
        ''' Split email address array strings into usable arrays '''
        split_embeds = lambda x: x.replace('[','').replace(']','').replace('\'','').split(',')
        addrs = [split_embeds(s) for s in tqdm(df[type].values)]
        u_addrs = [[self._address_clean(y) for y in x] for x in tqdm(addrs)]
        df[type] = u_addrs
        return df

    def _create_sentiment_dictionary(self, data_dir, negative_fn, positive_fn):
        '''
            Using sentiment lexicon from http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html

            Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
            Proceedings of the ACM SIGKDD International Conference on Knowledge 
            Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, Washington, USA, 
        '''
        negative_df = pd.read_csv(data_dir+negative_fn, names=['term'], header=None, comment=';')
        positive_df = pd.read_csv(data_dir+positive_fn, names=['term'], header=None, comment=';')

        sentiment = {x:'neg' for x in negative_df['term']}
        sentiment.update({x:'pos' for x in positive_df['term']})
        return sentiment

    def cliporpad(self, text:str, clen):
        ''' Just like it sounds.'''
        return text.ljust(clen)[0:clen]

    ###########################
    # Sentiment Classification 
    ###########################
    def _classify_sentiment(self, model, features, nbr_top_tokens=20, title='', mode=1):
        ''' Extract topics from LDA results and calculate sentiment labels '''

        print(f'\n--- Classify Sentiment - {title}\n\n')
        topics = []
        s_weighted_score = lambda a,s,t: sum([s[x] for x in range(len(a)) if a[x] == t])
        s_minmax_score = lambda a,min,max: ((a - a.min(axis=0)) / (a.max(axis=0) - a.min(axis=0))) * (max - min) + min
        s_stdize_score = lambda a: (a - a.mean(axis=0)) / a.std(axis=0)

        for idx, component in enumerate(model.components_): # iterate through all of the topics
            feature_index = component.argsort()[::-1]
            feature_names = [features[x] for x in feature_index]
            scores = [component[x] for x in feature_index]

            feature_len = nbr_top_tokens if nbr_top_tokens <= len(feature_index) else len(feature_index)

            subtopics = feature_names[0: feature_len]
            subscores = s_minmax_score(np.array(scores[0: feature_len]), 0.0, 50.0) # range bound scores
            if mode == 1:
                subsentiments = [self.sentiment_d[feature_names[x]] for x in range(feature_len)]
                sublabel_weighted_score = -1*s_weighted_score(subsentiments, subscores, 'neg') + s_weighted_score(subsentiments, subscores, 'pos') 
                sublabel_weighted = 'neg' if sublabel_weighted_score < -15.0 else 'pos' if sublabel_weighted_score > 15.0 else 'neu'
            elif mode == 2:
                terms = dict(map(lambda x: x, zip(subtopics, subscores)))
                sublabel_weighted_score = self.ac.calculate_sentiment_weighted(terms=terms)
                sublabel_weighted = 'neg' if sublabel_weighted_score < -0.5 else 'pos' if sublabel_weighted_score > 0.5 else 'neu'
            else:
                sublabel_weighted_score = 0.0
                sublabel_weighted = ''

            topic = {}
            topic['topics'] = subtopics
            topic['scores'] = subscores
            topic['weighted_score'] = sublabel_weighted_score
            topic['label'] = sublabel_weighted
            topics.append(topic)

            # display topic plus support subtopic words
            print(f'{self.cliporpad(str(idx), 3)} {topic["label"]} / {component[feature_index[0]]} = {" ".join(topic["topics"])}')

        # convert to dataframe
        df = pd.DataFrame(topics)
        print(f'\n{df.sort_values(by="label", ascending=False).head(200)}\n\n')

        # group to show overall sentiment totals
        print(f'{df[["label"]].groupby(by="label").size()}\n')

        return df

    def sentiment_analysis(self, macro_filter=0.8, micro_filter=10, n_components=100, mode=1):
        ''' Analyze email content using CV/LDA in preparation for sentiment calculation '''

        # create corpus from email content
        contents = self.email_df['Body'].to_list()

        # custom stop words - not needed if using a fixed vocabulary
        sw_list = None

        # fixed vocabulary of negative/positive keywords
        if mode == 1:
            vocab = {x:i for i,x in enumerate(self.sentiment_d.keys())}
        elif mode == 2:
            vocab = self.ac.get_sentiment_vocab()
        else:
            vocab = None

        # lemmatize if not using fixed vocabulary of negative/positive keywords
        tokenizer = None

        # fetch counts of word phrases
        start = time()
        tfc = CountVectorizer(max_df=macro_filter, min_df=micro_filter, max_features=20000, strip_accents='unicode', analyzer='word', 
                              token_pattern=r"[a-zA-Z][a-z][a-z][a-z]+", ngram_range=(1,1), stop_words=sw_list, vocabulary=vocab,
                              tokenizer=tokenizer)
        tf = tfc.fit_transform(contents)
        tf_a = tf.toarray()
        tf_fns = np.array(tfc.get_feature_names())
        print(f'--- Email Content Analysis w/ Sentiment Vocab ({time()-start} seconds)\n')

        start = time()
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=3, learning_method='online', learning_offset=50.0, random_state=1).fit(tf)
        print(f'--- LDA Analysis ({n_components} components in {time()-start} seconds)\n')

        results = self._classify_sentiment(lda, tf_fns, nbr_top_tokens=15, title='CV - LDA Model Sentiment '+str(mode), mode=mode)

        # add class label to training dataframe
        column = 'Class_Sentiment_'+str(mode)
        self._add_label_to_dataframe(lda.transform(tf), results, ('label',column))
        print(f'\n{self.email_df[[column]].groupby(by=column).size()}\n')

        return

    ####################################
    # Sentiment Classification w/ VADER
    ####################################
    def sentiment_analysis_vader(self):
        '''
            Using VADER from https://www.nltk.org/api/nltk.sentiment.html?highlight=vader#module-nltk.sentiment.vader

            Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. 
            Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
        '''
        # analyze and create supervised training dataset from email content
        vsa = vader.SentimentIntensityAnalyzer()

        start = time()
        vader_df = pd.DataFrame(self.email_df['Body'].apply(vsa.polarity_scores).to_list())
        vader_df['body'] = self.email_df['Body']
        #vader_df['dt'] = self.email_df['DateTime'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z'))
        vader_df['dt_y_m'] = self.email_df['DateTime'].apply(lambda x: (x.year*100)+x.month)
        print(f'\nVader Polarity Scoring ({time()-start} seconds)\n')

        print(f'\nVader Dataframe Statistics\n{vader_df.describe()}\n')

        sums = vader_df[['neg','neu','pos']].sum()
        print(f'\nVader Sentiment Totals\n{sums.head(20)}\n')

        vader_df['compound'].hist(bins=10); plt.show()
        vader_df.boxplot(column='compound', by='dt_y_m'); plt.show()
        vader_df['compound'].plot.kde(); plt.show()

        # add vader sentiment class to dataframe
        self.email_df['Class_Sentiment_Vader'] = vader_df['compound'].apply(lambda x: 'neg' if x < -0.20 else 'pos' if x > 0.20 else 'neu')

        return vader_df

    ##############################
    # Topic Classification Method 
    ##############################

    def _classify_topic_method(self, model, features, nbr_top_tokens=20, title='', classes=[], method='prob'):
        ''' Classify topics based upon subtopic prevalence and cosine similarity. '''

        print(f'\n--- Classify Topic Words Method - {title}\n\n')
        topics = []
        for idx, component in enumerate(model.components_): # iterate through all of the topics
            feature_index = component.argsort()[::-1]
            feature_names = [features[x] for x in feature_index]
            scores = [component[x] for x in feature_index]

            subtopics = []
            subtopics_scores = []
            for x in range(0, nbr_top_tokens): # iterate through the number of requested subtopics and calculate sentiment scores
                if x < len(feature_index): 
                    subtopics.append(feature_names[x])
                    subtopics_scores.append(scores[x])

            topic = {}
            topic['topics'] = subtopics
            topic['topics_scores'] = subtopics_scores

            # find the class label for these topic terms
            terms = {subtopics[x]:subtopics_scores[x] for x in range(len(subtopics))}
            label = self.ac.classify_terms(classes=classes, terms=terms, method=method, use_weights=True if method != 'none' else False) # use AutoClassify to determine label
            topic['label'] = label

            # save
            topics.append(topic)

            # display topic plus support subtopic words
            print(f'{self.cliporpad(str(idx), 3)} {self.cliporpad(label, 15)} / {subtopics_scores[0]} = {" ".join(subtopics)}')

        # convert to dataframe and save for analysis
        df = pd.DataFrame(topics)
        columns = ['label','topics']
        print(f'\n{df[columns].sort_values(by="label", ascending=False).head(200)}\n\n')
        df.to_csv(self.data_dir+self.topic_gradients_fn, index=False)

        # group to show overall sentiment totals
        print(f'{df[["label"]].groupby(by="label").size()}\n')
        return df

    def _stop_word_support_function(self, macro_filter=0.5, micro_filter=10):
        ''' Routine to collect word count list and pos word filter list for developing stop word vocabulary'''

        print(f'\n--- Stop Word Support Function Start')
        start = time()

        # create corpus from email content
        contents = self.email_df['Body'].to_list()
        vocab = None

        sw_list = [name.lower() for name in names.words()]
        sw_list.extend([stopword.lower() for stopword in stopwords.words()])

        tokenizer = self._body_content_analysis_tokenizer

        # fetch counts of word phrases
        tfc = CountVectorizer(max_df=macro_filter, min_df=micro_filter, max_features=20000, strip_accents='unicode', analyzer='word', 
                              token_pattern=r"[a-zA-Z][a-z][a-z][a-z]+", ngram_range=(1,1), stop_words=sw_list, vocabulary=vocab,
                              tokenizer=tokenizer)
        tf = tfc.fit_transform(contents)
        tf_a = tf.toarray()
        tf_fns = np.array(tfc.get_feature_names())

        # form/sort topic/sub-topic dataframe and save to file for manual analysis due to high variability
        # will use manual inspection to develop an algorithm for filtering list to a functional vocabulary
        sums = np.sum(tf_a, axis=0)
        dense_word_matrix = []
        exclude_word_matrix = ['wa']
        word_filter = lambda word: pos_tag([word])[0][1][0:2] not in ['NN','VB']
        word_list = words.words()

        for x in tqdm(range(len(sums))):
            phrase = {'phrase':tf_fns[x], 'count':sums[x]}
            dense_word_matrix.append(phrase)

            # collect words to filter - add to stop_words
            oov = True if tf_fns[x] not in word_list else False
            try:
                if oov or word_filter(tf_fns[x]):
                    exclude_word_matrix.append(tf_fns[x])
            except Exception as err:
                exclude_word_matrix.append(tf_fns[x])

        print(f'\n--- Stop Word Support Function Complete in ({time()-start} seconds)\n')

        sums_df = pd.DataFrame(dense_word_matrix).sort_values(by='count',ascending=False).reset_index(drop=True)
        sums_df.to_csv(self.data_dir+self.manual_word_counts_fn, index=False) # save to file for manual inspection
        print(f'\n--- Word Matrix\n\n{sums_df.head(20)}')

        word_filter_df = pd.DataFrame(exclude_word_matrix, columns=['word']).sort_values(by='word').reset_index(drop=True)
        word_filter_df.to_csv(self.data_dir+self.word_filter_list_fn, index=False) # save to file for manual inspection
        print(f'\n--- Word Filter\n\n{word_filter_df.head(20)}')

        return word_filter_df['word'].to_list()

    def _add_label_to_dataframe(self, dcmp, topics, columns):
        ''' Using the unsupervised sentiment analysis data, create a supervised learning dataset '''

        assert len(dcmp) == len(self.email_df), 'Length of decomposition matrix should match email dataframe length'

        values = []
        for x in tqdm(range(len(dcmp))):
            tidx = np.argmax(dcmp[x]) if np.argmax(dcmp[x]) > np.argmin(dcmp[x]) else -1
            value = topics.at[tidx, columns[0]] if tidx >= 0 else 'unknown'
            values.append(value)

        self.email_df[columns[1]] = pd.Series(values)
        return

    def _body_content_analysis_tokenizer(self, text, max_length=20):
        ''' CountVectorizer support function to tokenize and lemmatize incoming words.'''
        words = re.findall(r"[a-zA-Z][a-z][a-z][a-z]+", text)
        arr = [self.lemmatizer.lemmatize(w) for w in words if len(w) <= max_length] # could also use spacy here
        return arr

    def topic_classification(self, macro_filter=0.5, micro_filter=10, vectorizer='CountVectorizer', decomposition='LatentDirichletAllocation', classes=[], n_components=100, subtopics=15, method='prob', mode=1):
        '''
            General topic classification using various techniques.

            Content frequency distributions -> CountVectorizer & TFIDFVectorizer 
            Decomposition ->
                LDA - LatentDirichletAllocation
                LSA - TruncatedSVD
                NMF - Non-Negative Matrix Factorization

            Classification ->
                AutoClassify (developed by Avemac Systems LLC)
        '''

        # create corpus from email content
        contents = self.email_df['Body'].to_list()

        # custom stop words - not needed if using a fixed vocabulary
        sw_list = [name.lower() for name in names.words()]
        sw_list.extend([stopword.lower() for stopword in stopwords.words()])
        sw_list.extend(self._stop_word_support_function(macro_filter=macro_filter, micro_filter=micro_filter))
        sw_list.extend(['pirnie','skean','sithe','staab','montjoy','lawner','brawner']) # a few names that made it through the filters

        # fixed vocabulary of keywords
        vocab = None

        # lemmatizer
        tokenizer = self._body_content_analysis_tokenizer

        # fetch counts of word phrases
        print(f'\n--- Starting Email Content Analysis')

        start = time()

        if vectorizer == 'CountVectorizer':
            tfc = CountVectorizer(max_df=macro_filter, min_df=micro_filter, max_features=20000, strip_accents='unicode', analyzer='word', 
                                 token_pattern=r"[a-zA-Z][a-z][a-z][a-z]+", ngram_range=(1,1), stop_words=sw_list, vocabulary=vocab, tokenizer=tokenizer)
            title = 'Count'
        else:
            tfc = TfidfVectorizer(max_df=macro_filter, min_df=micro_filter, max_features=20000, strip_accents='unicode', analyzer='word', 
                                 token_pattern=r"[a-zA-Z][a-z][a-z][a-z]+", ngram_range=(1,1), stop_words=sw_list, vocabulary=vocab, tokenizer=tokenizer,
                                 use_idf=1, smooth_idf=1, sublinear_tf=1)
            title = 'TFIDF'

        tf = tfc.fit_transform(contents)
        tf_a = tf.toarray()
        tf_fns = np.array(tfc.get_feature_names())
        print(f'--- Content Frequency Analysis ({time()-start} seconds)\n')

        start = time()

        if decomposition == 'LatentDirichletAllocation':
            dcmp = LatentDirichletAllocation(n_components=n_components, max_iter=3, learning_method='online', learning_offset=50.0, random_state=1).fit(tf)
            title += ' - LDA'
        elif decomposition == 'TruncatedSVD':
            dcmp = TruncatedSVD(n_components=n_components, n_iter=100, random_state=1).fit(tf)
            title += ' - LSA'
        else:
            dcmp = NMF(n_components=n_components, random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=0.1, l1_ratio=0.5).fit(tf)
            title += ' - NMF'

        print(f'--- Decomposition Analysis ({n_components} components in {time()-start} seconds)\n')

        results = self._classify_topic_method(dcmp, tf_fns, nbr_top_tokens=subtopics, title=title, classes=classes, method=method)

        # add class label to training dataframe
        column = 'Class_Alignment_'+str(mode)
        self._add_label_to_dataframe(dcmp.transform(tf), results, ('label',column))
        print(f'\n{self.email_df[[column]].groupby(by=column).size()}\n')
        return


#####################################################################
# Main
#####################################################################

config = {
    'email_extracted_fn': 'extracted_emails.pd',
    'data_dir': '/proto/learning/avemac/email_analysis_blog/data/',
    'plot_image_save_directory': '/proto/learning/avemac/email_analysis_blog/plots/',
    'custom_stop_words_fn': 'custom_stop_words.txt',
    'negative_sentiment_fn': 'negative-words.txt',
    'positive_sentiment_fn': 'positive-words.txt',
    'sentiment_fn': 'sentiment.txt',
    'supervised_dataset_fn': 'supervised_email_train.csv',
    'word_filter_list_fn': 'word_filter_list.csv',
    'manual_word_counts_fn': 'content_word_counts.csv',
    'topic_gradients_fn': 'topic_gradients.csv',
}

usm = UnsupervisedModeling(config)

# Body content sentiment analysis method 1
'''
    Using sentiment lexicon from http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html

    Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
    Proceedings of the ACM SIGKDD International Conference on Knowledge 
    Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, Washington, USA, 
'''
x = usm.sentiment_analysis(mode=1)

# Body content sentiment analysis method 2
'''
    Using AFINN from http://corpustext.com/reference/sentiment_afinn.html

    Finn Årup Nielsen A new ANEW: Evaluation of a word list for sentiment analysis in microblogs.
    Proceedings of the ESWC2011 Workshop on 'Making Sense of Microposts': Big things come in small packages 718 in CEUR Workshop 
    Proceedings 93-98. 2011 May.

    Using AutoClassify for automatic topic labeling (Developed by Avemac Systems LLC)
'''
x = usm.sentiment_analysis(mode=2)

# Body content sentiment analysis with Vader
x = usm.sentiment_analysis_vader()

# Body content analysis - CountVectorize/LDA with AutoClassify
x = usm.topic_classification(macro_filter=0.5, vectorizer='CountVectorizer', decomposition='LatentDirichletAllocation', classes=['fun','work'], n_components=100, subtopics=20, method='softmax', mode=1)

# Body content analysis - CountVectorizer/NMF with AutoClassify
x = usm.topic_classification(macro_filter=0.5, vectorizer='CountVectorizer', decomposition='NMF', classes=['fun','work'], n_components=200, subtopics=20, method='softmax', mode=2)

# Body content analysis - TFIDF/NMF with AutoClassify
x = usm.topic_classification(macro_filter=0.5, vectorizer='TfidfVectorizer', decomposition='NMF', classes=['fun','work'], n_components=200, subtopics=20, method='softmax', mode=3)

##################
# Post Processing
##################

# Save resulting dataframe for later supervised modeling
usm.email_df.to_csv(config['data_dir']+config['supervised_dataset_fn'], index=False)

# Aggregate view
print(f'\n---Aggregate Class Results')
for column in [x for x in usm.email_df.columns if 'Class_' in x]:
    agg = usm.email_df[[column]].groupby(by=column).size().to_dict()
    print(f'{usm.cliporpad(column, 25)} {", ".join("=".join((k,str(v))) for (k,v) in agg.items())}')
print(f'\n')

exit()