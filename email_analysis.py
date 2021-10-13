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
# This source contains data analysis functions to validate the usability of the Enron
# email repository for the exercise of classifying sentiment and professional alignment
# in an unsupervised manner.
#
# -- This is the supporting code for my email insights and analysis blog series available
# on my website at https://www.avemacconsulting.com/resources/.  Part 2 of the series at
# https://www.avemacconsulting.com/2021/08/27/email-insights-from-data-science-part-2/
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
#  class BasicEmailStats - Contains all analysis methods.
#
# ---- Methods ----
#
#  def xy_gen_plot
#  def df_plot
#
#  def _address_clean - Regex to filter invalid email addresses
#  def _fix_email_addresses - Control function to correct invalid email addresses
#  def cliporpad - simple text formatting function
#
#  def unique_address_count - shows overall dataset counts by unique address and type
#  def _frequency_addresses_groupby - supporting routine for summarizing email address frequencies
#  def _frequency_addresses_groupby_sum - supporting routine for summarizing email address frequencies
#  def side_by_side_histogram - supporting routine for visualizing email address frequencies
#  def frequency_addresses - routine to summarize email frequencies by type and action
#  def frequency_subject_line - method to show the frequency distribution of tokens in all subject lines
#  def frequency_actions - frequency distribution for email actions (i.e. sent, received, deleted, etc.)
#  def frequency_time - distribution of email over time.
#  def frequency_day_of_week - distribution of emails by day of time.
#
#  def length_body_tokens - clustering emails by content length.
#  def graph_from_to_addresses - graphing email address relationships.
#  def datetime_zscore - datatime variability test.
#  def range_date_time - total date/time range for the dataset.
#  def manual_document_word_count - quick check of unique vocabulary size and word frequency.
#
#  def _multicollinearity_transform - support function for VIF test
#  def multicollinearity_test - VIF and correlation matrix tests
#
# ---- Main ----
#  Processing for all content and feature analysis routines
#
########################################################################################

#!/usr/bin/python3 -W ignore::DeprecationWarning
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import string
from time import time
import datetime as dt
import pandas as pd
import numpy as np
import collections
import re
from tqdm import tqdm
import sklearn.cluster
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_colwidth', 100)

#####################################################################
# Statistics and Analysis Functions
#####################################################################

class BasicEmailStats():
    ''' Generate simple statistics about email dataset'''

    def __init__(self, config):
        self.data_dir = config['data_dir']
        self.manual_word_counts_fn = config['manual_word_counts_fn']
        self.plot_save_dir = config['plot_image_save_directory']
        raw_emails = pd.read_csv(self.data_dir + config['email_extracted_fn'])
        raw_emails.fillna(value="[]", inplace=True)

        self.email_df = self._fix_email_addresses('From_Address', raw_emails)
        self.email_df = self._fix_email_addresses('To_Address', raw_emails)
        self.email_df = self._fix_email_addresses('Cc_Address', raw_emails)
        self.email_df = self._fix_email_addresses('Bcc_Address', raw_emails)

        self.email_df['DateTime'] = self.email_df['DateTime'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z'))
        self.email_df['DateTime_TS'] = self.email_df['DateTime'].apply(lambda x: x.timestamp())
        self.email_df['DateTime_HOUR'] = self.email_df['DateTime'].apply(lambda x: x.hour)
        self.email_df['DateTime_MONTH'] = self.email_df['DateTime'].apply(lambda x: x.month)

        print(f'\n--- Init Complete\n\n')

        return

    def cliporpad(self, text:str, clen):
        ''' Just like it sounds.'''
        return text.ljust(clen)[0:clen]

    def xy_gen_plot(self, X, X_label, Y, Y_label, title=None, aug_plots=None, aug_plot_labels=None, spot=None, spot_label=None, save=False, img_fn='generic_plot.png'):
        '''Generic graph plot - support X, Y and augmented plot points'''

        fig, (p1) = plt.subplots(1, 1)

        if title is not None:
            p1.set_title(title)

        p1.set_ylim([min(Y)-(max(Y)*0.1), max(Y)+(max(Y)*0.1)])
        p1.set_xlim([min(X)-(max(X)*0.1), max(X)+(max(X)*0.1)])
        p1.plot(X, Y, 'o-', color='blue')

        if aug_plots is not None:
            for x in range(len(aug_plots)):
                p1.plot(aug_plots[x][0], aug_plots[x][1], label=aug_plot_labels[x])

        if spot is not None:
            p1.plot(spot[0], spot[1], 'bo')
            p1.annotate(spot_label, xy=(spot[0]+0.2, spot[1]-0.2))

        p1.set_xlabel(X_label)
        p1.set_ylabel(Y_label)
        p1.legend()
        p1.grid(True)
        fig.tight_layout()

        if save: 
            plt.savefig(self.plot_save_dir + img_fn, format='png')

        plt.show()
        return

    def df_plot(self, ds, columns, types=['hist'], bins=20):
        '''Pandas-based plots'''
        for t in types:
            if t == 'box':
                ds.boxplot(column=columns)
            elif t == 'hist':
                ds.hist(column=columns, bins=bins)
            elif t == 'density':
                ds.plot.kde()
        plt.show()
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

    def unique_address_count(self):
        ''' Find unique email address counts '''
        from_addrs = list(set([y for x in self.email_df['From_Address'] for y in x]))
        to_addrs = list(set([y for x in self.email_df['To_Address'] for y in x]))
        cc_addrs = list(set([y for x in self.email_df['Cc_Address'] for y in x]))
        bcc_addrs = list(set([y for x in self.email_df['Bcc_Address'] for y in x]))

        addrs_df = pd.DataFrame([len(self.email_df)], columns=['Total_Emails'])
        addrs_df['From_Addresses'] = [len(from_addrs)]
        addrs_df['To_Addresses'] = [len(to_addrs)]
        addrs_df['Cc_Addresses'] = [len(cc_addrs)]
        addrs_df['Bcc_Addresses'] = [len(bcc_addrs)]

        print(f'\n--- Unique Email Address Counts\n\n{addrs_df.head(20)}\n')

        return addrs_df
    
    def _frequency_addresses_groupby(self, type):
        ''' Routine to summarize by email address and other features.'''

        columns = [type, 'Source','Day','Outside_Hours','Forwarded']
        # using list comprehension to build address dataframe from each address array within each email
        addrs = pd.DataFrame([[y, x[1], x[2], x[3], x[4]] for x in self.email_df[columns].values for y in x[0]], columns=columns)

        group = addrs.groupby(by=[type]) # group by address only so all data elements are captured within one reference point

        # since the grouping is by address type and we're looking for counts of multiple data elements
        # we'll just spin through the group details and manually sum each data element by the element type

        frequencies = []
        for grp in tqdm(group.groups):
            details = group.get_group(grp)
            total = len(details)
            sources = collections.Counter(details.Source.values)
            days = collections.Counter(details.Day.values)
            hours = collections.Counter(details.Outside_Hours.values)
            forwards = collections.Counter(details.Forwarded.values)

            # build a cum row for each group 
            frequency = {}
            frequency['user'] = grp
            frequency['total'] = total
            for k,v in sources.items(): frequency['sources_'+k] = v
            for k,v in days.items(): frequency['days_'+str(k)] = v
            for k,v in hours.items(): frequency['after_hours_'+str(k).lower()] = v
            for k,v in forwards.items(): frequency['forwards_'+str(k).lower()] = v
            frequencies.append(frequency)

        df = pd.DataFrame(frequencies).fillna(0.0).sort_values(by='total', ascending=False)
        return df[sorted(df.columns, reverse=True)]

    def _frequency_addresses_groupby_sum(self, group_details):
        ''' Anonymize the email address frequency data into a dataset summary.'''
        columns = sorted(group_details.columns, reverse=True)
        columns.remove('user')
        sums = {c:group_details[c].sum() for c in columns}
        return pd.DataFrame([sums])

    def side_by_side_histogram(self, df, bins=30):
        ''' Comparison plots. '''
        fig, axs = plt.subplots(1, 4, sharey=True, tight_layout=True)
        fig.text(0.5, 0.01, 'email count', ha='center')
        fig.text(0.01, 0.5, 'total users', va='center', rotation='vertical')
        axs[0].hist(df['total'], bins=bins); axs[0].title.set_text('total')
        axs[1].hist(df['sources_sent'], bins=bins); axs[1].title.set_text('sources_sent')
        axs[2].hist(df['sources_responded'], bins=bins); axs[2].title.set_text('sources_responded')
        axs[3].hist(df['sources_deleted'], bins=bins); axs[3].title.set_text('sources_deleted')
        plt.show()
        return

    def frequency_addresses(self):
        ''' Find unique email address frequency by type, action, etc. '''
        from_addrs_grp = self._frequency_addresses_groupby('From_Address')
        from_addrs_grp_sum = self._frequency_addresses_groupby_sum(from_addrs_grp)
        print(f'\n--- From Email Address Counts\n\n{from_addrs_grp.head(20)}\n\n{from_addrs_grp_sum.head(20)}')
        self.side_by_side_histogram(from_addrs_grp)

        to_addrs_grp = self._frequency_addresses_groupby('To_Address')
        to_addrs_grp_sum = self._frequency_addresses_groupby_sum(to_addrs_grp)
        print(f'\n--- To Email Address Counts\n\n{to_addrs_grp.head(20)}\n\n{to_addrs_grp_sum.head(20)}')
        self.side_by_side_histogram(to_addrs_grp)

        cc_addrs_grp = self._frequency_addresses_groupby('Cc_Address')
        cc_addrs_grp_sum = self._frequency_addresses_groupby_sum(cc_addrs_grp)
        print(f'\n--- Cc Email Address Counts\n\n{cc_addrs_grp.head(20)}\n\n{cc_addrs_grp_sum.head(20)}')
        self.side_by_side_histogram(cc_addrs_grp)

        return
    
    def frequency_subject_line(self):
        ''' Find subject line word frequency - Could use CountVectorizer or simple dictionary for this as well '''
        word_map = []
        for x in tqdm(self.email_df['Subject']):
            tokens = x.split()
            for token in tokens:
                if token not in ['Re:','RE:','FW:','-','[]','for','of','and','to','on','the','in','&']:
                    word_map.append({'word':token})
        word_map = pd.DataFrame(word_map)
        group = word_map.groupby(by=['word'])
        counts = group.size().to_frame(name='count').sort_values(by='count', ascending=False)
        print(f'\n--- Subject Line Token Frequency\n\n{counts.head(20)}')
        counts.hist(column='count', bins=100); plt.show()
        return

    def frequency_actions(self):
        ''' Email action frequency counts - using simple dictionary '''

        # define counts and probabilities
        df = self.email_df[['Source']]
        group = df.groupby('Source')
        actions = group.size()
        probs = actions.div(len(df))
        actions['P(sent)'] = probs['sent']
        actions['P(deleted)'] = probs['deleted']
        actions['P(responded)'] = probs['responded']
        actions['P(deleted|sent)'] = actions['P(deleted)'] * actions['P(sent)']
        actions['P(deleted|responded)'] = actions['P(deleted)'] * actions['P(responded)']
        actions['P(responded|sent)'] = actions['P(responded)'] * actions['P(sent)']
        print(f'\n--- Email Action Frequency\n\n{actions.to_frame().T.head(20)}')

        return

    def frequency_time(self):
        ''' Email frequency by hour of day '''
        group = self.email_df.groupby('DateTime_HOUR')
        counts = group.size()
        print(f'\n--- Email By Hour of Day Frequency\n\n{counts.to_frame().T.head(20)}')

        self.df_plot(self.email_df, columns=['DateTime_HOUR'], types=['box','hist'], bins=24)
        return

    def frequency_day_of_week(self):
        ''' Email frequency by day of week '''
        group = self.email_df[['Day']].groupby('Day')
        counts = group.size()
        print(f'\n--- Email By Day of Week Frequency\n\n{counts.to_frame().T.head(20)}')

        self.df_plot(self.email_df, columns=['Day'], types=['box','hist'], bins=7)
        return
        
    def length_body_tokens(self):
        ''' Cluster body content lengths - Use sklearn.cluster.KMeans '''
        lengths = []
        for x in tqdm(self.email_df['Body']):
            lengths.append([len(x)])
        estimator = sklearn.cluster.KMeans(n_clusters=20).fit(lengths)
        cluster_frequency = np.unique(estimator.labels_, return_counts=True)
        cluster_centers = [round(x[0],1) for x in estimator.cluster_centers_]
        df = pd.DataFrame([{'cluster_id':x,'cluster_count':cluster_frequency[1][x],'cluster_center':cluster_centers[x]} for x in cluster_frequency[0]]).sort_values(by='cluster_center')
        print(f'\n--- Body Content Segmentation By Character Length\n\n{df.head(20)}')

        self.xy_gen_plot(df.cluster_count, 'Email Group Count', df.cluster_center, 'Email Length (Center)', title='Email Body Content Length')
        return

    def graph_from_to_addresses(self):
        ''' Routine to generate a graph of email address relationships '''
        graph = nx.DiGraph()
        for from_a,to_arr in tqdm(self.email_df[['From_Address','To_Address']].values):

            if not graph.has_node(from_a[0]):
                graph.add_node(from_a[0])

            for to_a in to_arr:
                if not graph.has_node(to_a):
                    graph.add_node(to_a)
                if not graph.has_edge(from_a[0], to_a):
                    graph.add_edge(from_a[0], to_a, count=1)
                else:
                    graph.edges[from_a[0], to_a]['count'] = graph.edges[from_a[0], to_a]['count'] + 1

        nx.write_graphml(graph, self.plot_save_dir + 'from_to_addresses.graphml')
        #nx.draw_circular(graph)
        print(f'\n--- Graph From Addresses - To Addresses Info\n\n{nx.info(graph)}')
        print(f'\n--- Graph From Addresses - To Addresses Density\n\n{nx.density(graph)}')
        print(f'\n--- Graph From Addresses - To Addresses Degrees\n\n{nx.degree_histogram(graph)}')

        plt.show()

        return

    def datetime_zscore(self):
        ''' Quick check of probability for email date/time range '''
        zscores = self.email_df['DateTime_TS'].to_frame().apply(zscore)
        print(f'\n--- DateTime Z-Score, Oldest is {zscores["DateTime_TS"].max()} and Newest is {zscores["DateTime_TS"].min()}\n\n')
        zscores['DateTime_TS'].hist(bins=20); plt.show()
        zscores.boxplot(column='DateTime_TS'); plt.show()
        return

    def range_date_time(self):
        ''' Graph the email date/time range '''
        print(f'\n--- DateTime Min \'{self.email_df.DateTime.min()}\' & Max \'{self.email_df.DateTime.max()}\' Range\n\n')
        self.email_df['DateTime'].hist(bins=20); plt.show()
        return

    def manual_document_word_count(self):
        ''' Routine to investigate email content token frequencies '''
        words = {}
        punct_pattern = re.compile("[" + re.escape(string.punctuation) + "0-9" + "]")
        for x in tqdm(self.email_df['Body']):
            for y in re.sub(punct_pattern, "", x).lower().split(' '):
                count = words[y]+1 if y in words.keys() else 1
                words.update({y:count})
        print(f'\n--- Raw Word Count - dictionary len is {len(words)}, min count of {min(words.values())}, max count of {max(words.values())}\n')

        words = dict(filter(lambda item: item[1] > 5 and item[1] < 1000, words.items())) # roughly trim outliers
        print(f'\n--- Trimmed Word Count - dictionary len is {len(words)}, min count of {min(words.values())}, max count of {max(words.values())}\n')
        values = list(words.values())

        self.xy_gen_plot(np.arange(start=0,stop=len(words), dtype=int), 'vocab', sorted(values), 'count')
        self.df_plot(pd.DataFrame(values, columns=['count']), columns='count', bins=10)
        return words

    def _multicollinearity_transform(self, x):
        ''' Supporting function to encode data for VIF algorithm.'''
        if x.name == 'Outside_Hours':
            x = x.apply(lambda n: 1 if n else 0)
        elif x.name == 'Source':
            amap = {'deleted':0,'responded':1,'sent':2,'received':3}
            x = x.apply(lambda n: amap[n])
        elif x.name == 'Forwarded':
            x = x.apply(lambda n: 1 if n else 0)

        return x

    def multicollinearity_test(self):
        ''' Check for intervariable dependence. '''

        # vif check
        columns = ['Day','DateTime_HOUR','DateTime_MONTH','Outside_Hours','Source','Forwarded']
        vif_df = self.email_df[columns].apply(self._multicollinearity_transform)

        vdf = pd.DataFrame()
        vdf['features'] = columns
        vdf['vif'] = [variance_inflation_factor(vif_df.values, i) for i in range(len(vif_df.columns))]
        print(f'\n--- Variance Inflation Factor\n{vdf}\n')

        # correlation matrix and heatmap
        print(f'\n--- Correlation Matrix\n{vif_df.corr()}\n')
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(bottom=0.3,left=0.3)
        axs = sns.heatmap(vif_df.corr(), ax=ax); plt.show()

        return


#####################################################################
# Main
#####################################################################

config = {
    'email_extracted_fn': 'extracted_emails.pd',
    'data_dir': '/proto/learning/avemac/email_analysis_blog/data/',
    'plot_image_save_directory': '/proto/learning/avemac/email_analysis_blog/plots/',
    'manual_word_counts_fn': 'email_content_word_counts.csv',
}

ebs = BasicEmailStats(config)

x = ebs.unique_address_count()
x = ebs.range_date_time()
x = ebs.datetime_zscore()
x = ebs.frequency_addresses()
x = ebs.frequency_subject_line()
x = ebs.frequency_actions()
x = ebs.frequency_time()
x = ebs.frequency_day_of_week()
x = ebs.length_body_tokens()
x = ebs.manual_document_word_count()
x = ebs.graph_from_to_addresses()
x = ebs.multicollinearity_test()

exit()