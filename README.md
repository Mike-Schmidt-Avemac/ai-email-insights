
Supporting code repository for blog series on unsupervised feature/topic extraction from raw corporate email content.  

The dataset used for this exercise was specifically created from the raw Enron email repository located 
at **https://www.cs.cmu.edu/~enron/** with 3 labels generated for sentiment (positive/negative/neutral/unknown) 
and 3 labels for alignment(business/personal).  

Part 1 - https://www.avemacconsulting.com/2021/08/24/email-insights-from-data-science-techniques-part-1/
 
 Source file: email_extraction.py
 
 * **This source contains basic data filtering, cleaning and feature extraction methods
 to convert the raw Enron email repository into the initial input file for the data
 analysis steps in the pipeline.**

Part 2 - https://www.avemacconsulting.com/2021/08/27/email-insights-from-data-science-part-2/

 Source file: email_analysis.py
 
 * **This source contains data analysis functions to validate the usability of the Enron
 email repository for the exercise of classifying sentiment and professional alignment
 in an unsupervised manner.**
 
Part 3 - https://www.avemacconsulting.com/2021/09/23/email-insights-from-data-science-part-3/

 Source file: email_unsupervised.py
 
 * **This source contains content preparation and feature extraction/topic modeling routines
 for unsupervised data processing to generate a supervised classification output for
 subsequent model training.**

 * The code has been tested for topic modeling sentiment and professional alignment. Some
 feature extraction was also included, but not used during the model process.

 * Three different sentiment methods are included.  Two of the methods are based upon
 sentiment vocabularies and cosine similarity.  The remaining method is rule-based
 and utilizes the NLTK Vader algorithm. Classes include "negative", "positive", 
 "neutral" and "unknown".

 * Regarding professional alignment classification, three methods were also implemented.
 Two are based upon Non-Negative Matrix Factorization and the third utilizes
 Latent Dirichlet Allocation plus cosine similarity.  Class labels include "fun", "work"
 and "unknown".

Part 4 - https://www.avemacconsulting.com/2021/10/12/email-insights-from-data-science-part-4/
 
 Source file: email_supervised.py
 
 * **This source contains model preparation, training, evaluation and inference routines 
 to model supervised datasets for email classification tasks.**
 
 * The code has been tested for text tokenization to single label classification. Multi-label 
 and multi-feature support is built-in, but specific models will need to be developed to take 
 advantage of the framework.
 
 * Currently implements classification using recurrent networks (LSTM/GRU/RNN), Transformer 
 (encoder layer only), and prebuilt Roberta fine-tuning.
 
 * K-Fold cross validation is implemented...wouldn't take much to abstract the routines to accept 
 other CV methods.  CV can be turned off for a "blended" overfitting techique.
 
 * Ensemble evaluation logic has been implemented to support each model trained...plus a final 
 inference prediction output from an aggregated ensemble of all models.
 
**Python**

Python version 3.8.10 was used for development and testing.

**Package Requirements**

Packages used for development and testing include the following.  These may not be the minimum 
version acceptable; with the exception of torch v1.9.1 - that is a hard requirement.

$ pip list
Package                 Version            
----------------------- -------------------
matplotlib              3.4.2              
networkx                2.5.1              
nltk                    3.6.2                     
numpy                   1.21.2             
pandas                  1.2.4   
regex                   2021.4.4             
scikit-learn            0.24.2             
scipy                   1.7.1              
seaborn                 0.11.1               
torch                   1.9.1                          
tqdm                    4.49.0                 
transformers            4.6.1                 
wordnet                 0.0.1b2            
 
**Usage**

Development and testing on Ubuntu 20.04.  Should work fine in Windows environments as well. I did not create a script to 
run the applications in sequence so you'll need to create one yourself if needed.

1. Install all necessary python packages.
2. Modify the configuration in each source file to match the location on your workstation (Note. the apps will run on Kaggle also if made into notebooks.)
3. Download the Enron email repository (or use your own) and unzip.
4. Run email_extraction.py to generate the file 'extracted_emails.pd' in the data directory.
5. Run email_analysis.py to generate basic statistics and graphs of the dataset.
6. Download/install the sentiment lexicons from http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html and http://corpustext.com/reference/sentiment_afinn.html
7. Running email_unsupervised.py will generate the supervised training data file 'supervised_email_train.csv'.
8. Run email_supervised.py to create the classification models, performance graphs and vocabulary files.
