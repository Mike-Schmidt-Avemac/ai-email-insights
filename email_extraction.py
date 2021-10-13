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
# This source contains basic data filtering, cleaning and feature extraction methods
# to convert the raw Enron email repository into the initial input file for the data
# analysis steps in the pipeline.
#
# -- This is the supporting code for my email insights and analysis blog series available
# on my website at https://www.avemacconsulting.com/resources/.  Part 1 of the series at
# https://www.avemacconsulting.com/2021/08/24/email-insights-from-data-science-techniques-part-1/
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
#  class EmailExtraction - Contains all extraction methods.
#
# ---- Methods ----
#
# def _extract_user_name - parse the username from the physical account subfolder.
# def _determine_email_action - determine what state the email is in.
# def _clean - remove unwanted text from email body.
# def _is_external_origination - determine if email originated externally.
# def _is_system_generated - determine if email is system generated or not.
# def _parse_emails - main content management processing loop.
#
# ---- Main ----
#  Processing for all content filtering, cleaning and feature collection.
#
########################################################################################

#!/usr/bin/python3 -W ignore::DeprecationWarning
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import datetime as dt
import pandas as pd
import numpy as np
import glob
import re
import email.parser as ep
from tqdm import tqdm
from nltk.corpus import names

pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_colwidth', 100)

#####################################################################
# Email content extraction and filtering routines.
#####################################################################

class EmailExtraction():
    ''' Process raw emails into data frame'''

    def __init__(self, config):
        ''' Initialize structures and parse emails upon instantiation '''

        # create a proper names list to estimate system email addresses from actual user email addresses
        print('--- Process proper names list')
        self._name_list = [name.lower() for name in tqdm(names.words())]

        # parse emails, keeping only the information we're interested in
        print('--- Parse raw email data')
        self.email_data = self._parse_emails(config['email_base_dir'], limit=config['email_limit'])

        return 
    
    def _extract_user_name(self, email_fn):
        ''' Pulls the account user name from the physical directory structure '''
        segments = email_fn.split('/')
        return segments[-3]

    def _determine_email_action(self, email_fn):
        ''' Looking at the physical email location, determine the action '''
        action = None

        # is username in "to" or "from" address list
        if 'deleted_items' in email_fn:
            action = 'deleted'
        elif 'sent_items' in email_fn:
            action = 'responded'
        elif 'sent' in email_fn:
            action = 'sent'
        else:
            action = 'received'

        return action

    def _clean(self, payload):
        ''' Remove unwanted information from email body '''

        text = payload
        marks = ['`', '&', '*', '+', '/', '<', '=', '>', '[', '\\', ']', '^', '_', '{', '|', '}', '~', '»', '«'] 
        punct_pattern = re.compile("[" + re.escape("".join(marks)) + "]")

        text = re.sub(r'\n', " ", text) # remove newlines
        text = re.sub(r'\r', " ", text) # remove DOS returns
        text = re.sub(r'\t', " ", text) # replace tabs 
        text = re.sub(r'=[0-9][0-9]', '', text) # remove parsing artifacts
        text = re.sub(r'[^\040-\176]+', '', text) # remove invalid characters
        text = re.sub(punct_pattern, "", text) # remove unneeded punct

        text = re.sub(r'-----Original Message-----.*', '', text)
        text = re.sub(r'From: .*', '', text)
        text = re.sub(r'----- Forwarded by.*', '', text)
        text = re.sub(r'---------------------- Forwarded by.*', '', text)
        text = re.sub(r'--------- Inline attachment follows.*', '', text)
        text = re.sub(r'Start Date: [0-9]+.*', '', text)

        text = re.sub(r'  +', ' ', text) # cleanup whitespace
        return text

    def _is_external_origination(self, from_addr):
        ''' Determine if email originated external to the company '''
        addr_parts = from_addr.split('@')
        return False if len(addr_parts) == 2 and addr_parts[1].lower() == 'enron.com' else True

    def _is_system_generated(self, from_addr):
        ''' Determine if email was system generated. Note - doesn't work well with complex names '''
        parts = re.sub(r'@.*', '', from_addr).split('.')
        proper_nouns = [1 if x in self._name_list else 0 for x in parts]
        return True if sum(proper_nouns) == 0 else False

    def _parse_emails(self, base_dir, limit=sys.maxsize):
        ''' Loop through all of the email files and extract/infer features '''

        email_sources = glob.glob(base_dir + '*/deleted_items/*')
        email_sources.extend(glob.glob(base_dir + '*/sent_items/*'))
        email_sources.extend(glob.glob(base_dir + '*/sent/*'))

        parser = ep.Parser()
        emails = []

        stop = 0

        for email_fn in tqdm(email_sources[0:limit]):

            # retrieve email content
            try:
                with open(email_fn, 'r') as f:
                    email = parser.parse(f)
            except Exception:
                continue # encoder error, skip

            user_name = self._extract_user_name(email_fn)

            # skip external and system generated emails
            if self._is_external_origination(email.get('From')): continue
            if self._is_system_generated(email.get('From')): continue

            # extract fields
            fields = {}
            date_time = dt.datetime.strptime(email.get('Date')[:-6], '%a, %d %b %Y %H:%M:%S %z')
            fields['DateTime'] = date_time
            fields['Day'] = date_time.weekday()
            fields['Outside_Hours'] = date_time < dt.datetime(date_time.year, date_time.month, date_time.day, 7, 0, 0, tzinfo=dt.timezone(date_time.utcoffset())) or date_time > dt.datetime(date_time.year, date_time.month, date_time.day, 18, 0, 0, tzinfo=dt.timezone(date_time.utcoffset()))
            fields['From_Address'] = email.get('From')
            fields['To_Address'] = [x for x in email.get('To').replace('\n','').replace('\t','').split(',')] if email.get('To') is not None else None
            fields['Cc_Address'] = [x for x in email.get('Cc').replace('\n','').replace('\t','').split(',')] if email.get('Cc') is not None else None
            fields['Bcc_Address'] = [x for x in email.get('Bcc').replace('\n','').replace('\t','').split(',')] if email.get('Bcc') is not None else None
            fields['Subject'] = email.get('Subject')
            fields['Forwarded'] = 'Fwd' in email.get('Subject') or 'FW' in email.get('Subject') or 'Forwarded' in email.get_payload()
            fields['Source'] = self._determine_email_action(email_fn)
            fields['Body'] = self._clean(email.get_payload())

            if len(fields['Body']) <= 1: continue # skip empty emails

            emails.append(fields)

        # deduplicate content
        df = pd.DataFrame(emails).drop_duplicates(subset='Body').reset_index(drop=True)
        print('--- Found %d emails out of %d possible' % (len(df), limit))
            
        return df


#####################################################################
# Main
#####################################################################

config = {
    'email_base_dir': '/proto/learning/avemac/email_analysis_blog/data/maildir/',
    'email_limit': sys.maxsize,
    'email_extracted_fn': 'extracted_emails.pd',
    'data_dir': '/proto/learning/avemac/email_analysis_blog/data/',
}

email_df = EmailExtraction(config).email_data
email_df.to_csv(config['data_dir'] + config['email_extracted_fn']) #save to file for next step in pipeline
print('--- Sample parsed data')
print(email_df)
exit()