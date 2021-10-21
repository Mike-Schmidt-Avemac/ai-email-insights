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
# This source contains model preparation, training, evaluation and inference routines 
# to model supervised datasets for email classification tasks.
#
# -- This is the supporting code for my email insights and analysis blog series part 4 
# available on my website at https://www.avemacconsulting.com.  Part 4 of the series at
# https://www.avemacconsulting.com/2021/10/12/email-insights-from-data-science-part-4/
#
# The code has been tested for text tokenization to single label classification.  
# Multi-label and multi-feature support is built-in, but specific models will need to be
# developed to take advantage of the framework.
#
# Currently implements classification using recurrent networks (LSTM/GRU/RNN), 
# Transformer (encoder layer only), and prebuilt Roberta fine-tuning.
#
# K-Fold cross validation is implemented...wouldn't take much to abstract the routines 
# to accept other CV methods.  CV can be turned off for a "blended" overfitting techique.
#
# Ensemble evaluation logic has been implemented to support each model trained...plus 
# a final inference prediction output from an aggregated ensemble of all models.
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
#
# ---- Classes ----
#  class ContentDataset(Dataset) - Custom Pytorch "Dataset" implementation for tokenized email content.
#  class Vocabulary() - Class for saving / retrieving / generating custom vocabulary from email content.
#  class RawDataLoader() - Class methods for retrieving and formatting raw dataset into Pandas dataframe.
#  class ModelSupport() - Weight/Bias initialization and graphing routines.
#  class SupervisedRNN(nn.Module) - Pytorch recurrent model implementation (LSTM/GRU/RNN)
#  class SupervisedTransformer(nn.Module) - Pytorch TransformerEncoder model.
#  class PositionalEncoding(nn.Module) - SupervisedTransformer supporting function for positional embeddings.
#  class SupervisedPrebuilt(nn.Module) - HuggingFace Robert-Base prebuilt transformer model implementation.
#  class ModelManagement() - Common training/eval and state management routines for model creation.
#  class PipelineConfig() - Common configuration class for Training and Inference pipeline logic.
#  class TrainingPipeline() - Training/Eval pipeline logic.
#  class InferencePipeline() - Inference pipeline logic.
#
# ---- Main ----
#  Train/Eval Processing - Label and model selection point and main train/eval processing loop.
#  Inference Testing - Label and model selection point and main inference processing loop.
#
########################################################################################

#!/usr/bin/python3 -W ignore::DeprecationWarning
import os
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import re
import math
import glob
import pickle
import gc
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

from sklearn.model_selection import KFold

from transformers import RobertaConfig, RobertaTokenizerFast, RobertaModel
from nltk.corpus import stopwords

pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_colwidth', 100)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_KEY = '<pad>'
NBR_KEY = '<nbr>'

KAGGLE_BASEDIR = '/kaggle'
LOCAL_BASEDIR = '/proto/learning/avemac/email_analysis_blog'
LOCAL_PRETRAIN_BASEDIR = '/proto/models/hf'
IS_LOCAL = not os.path.exists(KAGGLE_BASEDIR) # running in Kaggle Environment or not


##############################################################################################################################
# Custom Content Dataset
##############################################################################################################################

class ContentDataset(Dataset):
    ''' 
        Custom Pytorch "Dataset" implementation for tokenized email content.
        Implement the PyTorch dataset functions - example at https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler

        - Requires label column be renamed to "label" if not already.
        - Requires text column be renamed to "content" if not already.
    '''
    def __init__(self, df:pd.DataFrame, config:dict, vocab:dict) -> None:
        super(ContentDataset).__init__()
        self.config = config

        print(f'\n--- Building ContentDataset for embedding_type "{config["embedding_type"]}"')
        
        # tokenization is different if using trained versus pretrained models
        if config['embedding_type'] == 'train':
            df['text'] = self._custom_text_encoder(df['content'].values, vocab, max_tokens=config['max_tokens'])
            columns = list(df.columns); columns.remove('content')
        elif config['embedding_type'] == 'hgf_pretrained':
            tokenizer = RobertaTokenizerFast.from_pretrained(f'{config["pretrained_dir"]}{config["pretrained_model"]}')
            t = tokenizer(df['content'].to_list(), add_special_tokens=True, return_attention_mask=True, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors='np')
            df['input_ids'] = t['input_ids'].tolist()
            df['attention_mask'] = t['attention_mask'].tolist()
            columns = ['input_ids','attention_mask']
            if 'label' in df.columns: columns.append('label')
        else:
            raise AssertionError('Invalid tokenization mode')

        self.tds = {x : torch.tensor(df[x].to_list()).view(len(df),-1) for x in columns}
        self.length = len(df)
        return

    def __iter__(self) -> iter:
        self.pos = 0
        return self

    def __next__(self) -> dict:
        if self.pos < self.__len__(): 
            slice = {key:tensor[self.pos] for key, tensor in self.tds.items()}
            self.pos += 1
            return slice
        else:
            raise StopIteration
    
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> tuple():
        return {key:tensor[index] for key, tensor in self.tds.items()}

    def _custom_text_encoder(self, corpus, vocab, max_tokens=500):
        ''' Encode raw text content into dense token vectors for future embedding layer input. '''
        encoded = []
        
        for text in tqdm(corpus):
            encode = []
            tokens = re.findall(r"[a-zA-Z][a-z'][a-z']+", text)
            for t in tokens:
                t = t.lower()
                if t in vocab: # only work with words in vocab
                    encode.append(vocab[t]) # encode token
                if len(encode) >= max_tokens: break # stop if beyond max tokens
            if len(encode) < max_tokens: # pad manually instead of using nn.utils.rnn.pad_packed_sequence in model
                encode.extend([vocab[PAD_KEY] for _ in range(len(encode), max_tokens)])
            encoded.append(encode) # add sample row
        
        return encoded


##############################################################################################################################
# Vocabulary Tokenizer
##############################################################################################################################

class Vocabulary():
    ''' Class for saving / retrieving / generating custom vocabulary from email content. '''
    def __init__(self, config) -> None:
        self.config = config
        self.vocab = None
        return

    def get_vocabulary(self, corpus=None, force_build=False) -> dict:
        ''' Retrieve the vocabulary if exists or generate new if forced or not found. '''
        if self.vocab == None:
            vocab_fn = f'{self.config["checkpoint_dir"]}{self.config["vocabulary_fn"]}'
            if os.path.exists(vocab_fn) and not force_build:
                with open(vocab_fn, 'rb') as f:
                    self.vocab = pickle.load(f)
            elif corpus is not None:
                self.vocab = self._create_vocabulary(corpus)
                with open(vocab_fn, 'wb') as f:
                    pickle.dump(self.vocab, f) 
        return self.vocab

    def _create_vocabulary(self, corpus) -> dict:
        ''' Iterate through the data samples and create the token vocabulary '''
        stop_words = [w.lower() for w in stopwords.words()]
        vocab = {PAD_KEY:0, NBR_KEY:1}; vocab_idx = 2
        
        for text in tqdm(corpus):
            tokens = re.findall(r"[a-zA-Z][a-z'][a-z']+", text)
            for t in tokens:
                t = t.lower()
                if len(t) > 20: continue # skip long "words", most likely not a real word
                if t in stop_words: continue # skip stopwords

                if t not in vocab: # update vocab if token missing
                    vocab[t] = vocab_idx
                    vocab_idx += 1
        return vocab


##############################################################################################################################
# Training and Eval Dataset Loader
##############################################################################################################################

class RawDataLoader():
    ''' Class methods for retrieving and formatting raw dataset into Pandas dataframe. '''
    def __init__(self, config) -> None:
        self.data_dir = config['data_dir']
        self.columns = config['input_columns'] # dictionary {<actual>:<renamed>}
        self.config = config

        self.class_encoders = { 
            # can also use sklearn.preprocessing.LabelEncoder or PyTorch LabelEncoder or others...
            # regardless of method, for non-labels the same encoding scheme must be used during model inference
            'Outside_Hours':lambda x: 0 if x is False else 1,
            'Forwarded':lambda x: 0 if x is False else 1,
            'Source':lambda x: 0 if x == 'deleted' else 1 if x == 'responded' else 2 if x == 'sent' else 3,
            'Class_Alignment_1':lambda x: 0 if x == 'work' else 1 if x == 'fun' else 2,
            'Class_Alignment_2':lambda x: 0 if x == 'work' else 1 if x == 'fun' else 2,
            'Class_Alignment_3':lambda x: 0 if x == 'work' else 1 if x == 'fun' else 2,
            'Class_Sentiment_1':lambda x: 0 if x == 'neg' else 1 if x == 'pos' else 2,
            'Class_Sentiment_2':lambda x: 0 if x == 'neg' else 1 if x == 'pos' else 2,
            'Class_Sentiment_Vader':lambda x: 0 if x == 'neg' else 1 if x == 'pos' else 2,
        }

        self.df = self._fetch_raw_data(self.data_dir + config['supervised_dataset_fn'], self.columns)
        self.vocab = Vocabulary(self.config).get_vocabulary(corpus=self.df['content'].values, force_build=True)
        return

    def _fetch_raw_data(self, fn:str, columns:dict) -> pd.DataFrame:
        ''' Routine to fetch the labeled training data, normalize some column names and encode classes'''
        limit = self.config['limit']
        keys = columns.keys()

        # retrieve input dataframe from unsupervised pipeline process - note: sample will auto shuffle so set state for test consistency
        raw_data = (pd.DataFrame)(pd.read_csv(fn)[keys]).sample(frac=limit, random_state=42)

        # encode class if included in the columns list
        for column in keys:
            if column in self.class_encoders.keys():
                raw_data[column] = raw_data[column].apply(self.class_encoders[column])

        # rename raw feature names to standard names
        raw_data.rename(columns=columns, inplace=True)
        return raw_data

    def split_dataset(self, df=None, split=0.1) -> tuple():
        ''' Break the dataset up based upon the split requested. '''
        df = self.df if df is None else df
        s2 = df.sample(frac=split, random_state=42)
        s1 = df.drop(s2.index)
        return (s1, s2) # return train/test


##############################################################################################################################
# Common Model Support Routines
##############################################################################################################################

class ModelSupport():
    ''' 
        Common model weight initialization and graphing support routines

        Note: uses recursion.  
        Known issues: init_weights Will not properly process nested container modules of the same type (i.e. a ModuleList directly inside a ModuleList)
    '''
    def __init__(self, config):
        self.config = config
        return

    def init_weights(self, module, mode='relu'):
        ''' Recurse through container modules and initialze weights/biases by module type. '''
        if isinstance(module, (nn.ModuleList, nn.ModuleDict, nn.Sequential, nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            for m in module.modules():
                _ = self.init_weights(m) if m != module else None
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            _ = nn.init.zeros_(module.bias) if module.bias is not None else None
        elif isinstance(module, (nn.Linear, nn.Embedding, nn.LSTM, nn.GRU, nn.MultiheadAttention, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            for name, param in module.named_parameters(): 
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    _ = nn.init.xavier_normal_(param) if mode=='tanh' else nn.init.kaiming_normal_(param)
        return

    def graph_outputs(self, fold, epoch, step, outputs:dict, mode=['hist'], layers=[]):
        ''' Debugging routine to visualze model layer outputs and detect anomalies. '''
        layers = layers if len(layers) > 0 else [k for k in outputs.keys()]
        subplot_rows = round(math.sqrt(len(layers)))
        subplot_cols = math.ceil(math.sqrt(len(layers)))

        # histogram
        if 'hist' in mode:
            idx = 0
            fig, axes1 = plt.subplots(subplot_rows, subplot_cols, figsize=(subplot_cols*5,subplot_rows*2))
            fig.suptitle('Output Tensor Distributions', y=0.99)
            fig.supylabel('Frequency')
            fig.subplots_adjust(top=0.90, bottom=0.1, wspace=0.35, hspace=0.80)
            axes1 = axes1.flatten()
            for name, output in outputs.items():
                if name in layers:
                    d = output.detach().cpu().numpy().flatten()
                    ax = axes1[idx]
                    ax.set_title(f'{name} - {d.size}', {'fontsize':8})
                    ax.tick_params(axis='both', which='major')
                    ax.hist(d, bins=100)
                    idx += 1
            fig.savefig(f'{self.config["graph_dir"]}Outputs_{self.config["model_id"]}_F{fold}E{epoch}S{step}.png', facecolor=fig.get_facecolor())
            plt.close(fig)
        return

    def graph_parameters(self, model, fold, epoch, step, mode=['hist'], types=['weight','bias'], module_types=(), spot_check=True):
        ''' Routine to plot layer weights and verify acceptable neural processing.

            Note -> using this routine for realtime analysis via debugger, could also save 
                    checkpoints every iteration and feed into TensorBoard.

            Should also create an abstract class for this at some point...
        '''
        if 'grad' in types and len(types) > 1:
            raise AssertionError('Cannot mix gradient and weights/bias on same graph')

        module_types = module_types if len(module_types) > 0 else (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.LSTM, nn.GRU, nn.TransformerEncoder, nn.TransformerDecoder)
        m_list = []
        m_count = 0

        # create list of modules to display, plus count the number of weight vectors for subplot matrix
        for m in model.modules():
            if isinstance(m, module_types):
                m_list.append(m)
                for name, param in m.named_parameters():
                    if 'weight' in name and 'weight' in types:
                        m_count += 4 if isinstance(m, nn.LSTM) else 3 if isinstance(m, nn.GRU) else 1
                    if 'bias' in name and 'bias' in types:
                        m_count += 1
                    if 'grad' in types and param.grad is not None:
                        m_count += 1

        # nothing to graph
        if m_count <= 0:
            return

        subplot_rows = round(math.sqrt(m_count))
        subplot_cols = math.ceil(math.sqrt(m_count))

        # histogram
        if 'hist' in mode:
            idx = 0
            fig, axes1 = plt.subplots(subplot_rows, subplot_cols, figsize=(subplot_cols*5,subplot_rows*2))
            fig.suptitle('Parameter Distributions' if 'grad' not in types else 'Gradient Distribution', y=0.99)
            fig.supylabel('Frequency')
            fig.subplots_adjust(top=0.90, bottom=0.1, wspace=0.35, hspace=0.80)
            axes1 = axes1.flatten()
            for m in m_list:
                for name, param in m.named_parameters():
                    if 'weight' in name and 'weight' in types:
                        if isinstance(m, nn.LSTM):
                            w_i, w_f, w_c, w_o = param.chunk(4, 0)
                            d = {
                                'w_i':w_i.detach().cpu().numpy().flatten(),
                                'w_f':w_f.detach().cpu().numpy().flatten(),
                                'w_c':w_c.detach().cpu().numpy().flatten(),
                                'w_o':w_o.detach().cpu().numpy().flatten(),
                            }
                        elif isinstance(m, nn.GRU):
                            w_r, w_i, w_n = param.chunk(3, 0)
                            d = {
                                'w_i':w_i.detach().cpu().numpy().flatten(),
                                'w_r':w_r.detach().cpu().numpy().flatten(),
                                'w_n':w_n.detach().cpu().numpy().flatten(),
                            }
                        else:
                            d = {
                                'w_h': param.data.detach().cpu().numpy().flatten()
                            }

                        for k,v in d.items():
                            ax = axes1[idx]
                            ax.set_title(f'{m._get_name()} - {name}/{k} - {v.size}', {'fontsize':8})
                            ax.tick_params(axis='both', which='major')
                            ax.hist(v, bins=100)
                            idx += 1

                    if 'bias' in name and 'bias' in types:
                        d = param.data.detach().cpu().numpy().flatten()
                        ax = axes1[idx]
                        ax.set_title(f'{m._get_name()} - {name} - {d.size}', {'fontsize':8})
                        ax.tick_params(axis='both', which='major')
                        ax.hist(d, bins=100)
                        idx += 1

                    if 'grad' in types and param.grad is not None:
                        d = param.grad.detach().cpu().numpy().flatten()
                        ax = axes1[idx]
                        ax.set_title(f'{m._get_name()} - {name}/grad - {d.size}', {'fontsize':8})
                        ax.tick_params(axis='both', which='major')
                        ax.hist(d, bins=100)
                        idx += 1

            fig.savefig(f'{self.config["graph_dir"]}{"Gradients" if "grad" in types else "Parameters"}_{self.config["model_id"]}_F{fold}E{epoch}S{step}_{"Check" if spot_check else "Debug"}.png', facecolor=fig.get_facecolor())
            plt.close(fig)
        return


##############################################################################################################################
# Supervised Recurrent Model
##############################################################################################################################

class SupervisedRNN(nn.Module):
    ''' Recurrent network for time-series model of email content '''

    def __init__(self, mode:str, config:dict):
        super().__init__()
        self.batch_size = config['batch_size'] # batch length
        self.max_tokens = config['max_tokens'] # sequence length
        self.embedding_len = config['embedding_len'] # feature length
        self.number_classes = config['number_classes'] # number target classes
        self.dropout = config['dropout']
        self.epochs = config['epochs']
        self.vocab_size = config['vocab_size']
        self.bidirectional = config['bidirectional']
        self.rnn_layers = config['rnn_layers']
        self.config = config

        # define network

        # Embedding layer - training custom token relationships rather than pretrained
        # Note: should train this separately 
        self.embedding = nn.ModuleList([
            nn.Embedding(self.vocab_size, self.embedding_len, scale_grad_by_freq=True),
        ])

        # Recurrent network (Input = BatchSize x MaxTokens x Embedding Length)
        ln_input_len = (2 if self.bidirectional else 1) * self.rnn_layers * self.embedding_len
        self.rnn = nn.ModuleList([
            nn.LSTM(self.embedding_len, self.embedding_len, batch_first=True, num_layers=self.rnn_layers, bidirectional=self.bidirectional) if mode=='lstm' 
            else nn.GRU(self.embedding_len, self.embedding_len, batch_first=True, num_layers=self.rnn_layers, bidirectional=self.bidirectional) if mode=='gru'
            else nn.RNN(self.embedding_len, self.embedding_len, batch_first=True, num_layers=self.rnn_layers, bidirectional=self.bidirectional),
            nn.LayerNorm(ln_input_len), # input size is doubled if birectional LSTM
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
        ])

        # Fully connected network to reduce recurrent weights to log odds
        fc_input_len = (2 if self.bidirectional else 1) * self.rnn_layers * self.embedding_len
        self.fc = nn.ModuleList([
            nn.Linear(fc_input_len, fc_input_len * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_input_len * 2, fc_input_len),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_input_len, self.number_classes),
            nn.LogSoftmax(),
        ])

        self.reset_weights(mode='init')

        self.to(DEVICE) # move to GPU if available
        return

    def forward(self, text, **kwargs):
        ''' Forward pass for recurrent network '''
        output_checks = {}
        output_pos = 0

        x = text

        # process the embedding layer
        for m in self.embedding:
            x = m(x)
            if self.config['check_outputs']:
                output_checks[f'{m.__class__.__name__}-Layer{output_pos}'] = x; output_pos += 1

        # spin through the modules defined within the recurrent network portion of the model
        for m in self.rnn:
            if isinstance(m, (nn.LSTM)):
                _, (x,_) = m(x) # use last hidden state since we are labeling the entire sequence
                x = torch.transpose(x, 0, 1) # back to batch first
                x = x.reshape(-1, np.prod(x.shape[-2:])) if x.ndim == 3 else x # collapse the sequence dimension if bidirectional
            elif isinstance(m, (nn.GRU)):
                _, x = m(x) # use last hidden state since we are labeling the entire sequence
                x = torch.transpose(x, 0, 1) # back to batch first
                x = x.reshape(-1, np.prod(x.shape[-2:])) if x.ndim == 3 else x # collapse the sequence dimension if bidirectional
            else:
                x = m(x)

            if self.config['check_outputs']:
                output_checks[f'{m.__class__.__name__}-Layer{output_pos}'] = x; output_pos += 1

        # finish up the pass regressing the recurrent summarized sequence weights into class log odds
        for m in self.fc:
            x = m(x)
            if self.config['check_outputs']:
                output_checks[f'{m.__class__.__name__}-Layer{output_pos}'] = x; output_pos += 1

        return x, output_checks

    def reset_weights(self, mode='fold'):
        ''' Method for initial weights and resetting weights between cross-validation folds '''
        minit = ModelSupport(self.config)
        _ = minit.init_weights(self.embedding) if mode == 'init' else None
        minit.init_weights(self.rnn)
        minit.init_weights(self.fc)
        return


##############################################################################################################################
# Supervised Transformer Model
##############################################################################################################################

class SupervisedTransformer(nn.Module):
    ''' Transformer network for time-series model of email content '''
    def __init__(self, config:dict):
        super().__init__()
        self.batch_size = config['batch_size'] # batch length
        self.max_tokens = config['max_tokens'] # sequence length
        self.embedding_len = config['embedding_len'] # feature length
        self.number_classes = config['number_classes'] # number target classes
        self.dropout = config['dropout']
        self.epochs = config['epochs']
        self.vocab_size = config['vocab_size']
        self.attention_heads = config['attention_heads']
        self.encoder_layers = config['encoder_layers']
        self.config = config

        # define network

        # Embedding layer - training custom token relationships rather than pretrained
        self.embedding = nn.ModuleList([
            nn.Embedding(self.vocab_size, self.embedding_len, scale_grad_by_freq=True),
        ])

        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(self.embedding_len, dropout=self.dropout, max_len=self.max_tokens, batch_first=True)

        # Encoder network (Input = BatchSize x MaxTokens x Embedding Length)
        attention_layer = nn.TransformerEncoderLayer(self.embedding_len, self.attention_heads, dim_feedforward=self.max_tokens*4, dropout=self.dropout, batch_first=True)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoder(attention_layer, self.encoder_layers),
            nn.LayerNorm(self.embedding_len),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        ])

        # Fully connected network to reduce encoder weights to log odds
        fc_input_len = self.max_tokens * self.embedding_len # 2d size after flatten of 3d transformer output
        self.fc = nn.ModuleList([
            nn.Linear(fc_input_len, fc_input_len // 4),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_input_len // 4, fc_input_len // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_input_len // 2, self.number_classes),
            nn.LogSoftmax(),
        ])

        self.reset_weights(mode='init')

        self.to(DEVICE) # move to GPU if available
        return

    def forward(self, text, **kwargs):
        ''' Forward pass for transformer network '''
        output_checks = {}
        output_pos = 0

        x = text

        # process the embedding layer
        for m in self.embedding:
            x = m(x)
            if self.config['check_outputs']:
                output_checks[f'{m.__class__.__name__}-Layer{output_pos}'] = x; output_pos += 1

        # encode with position information
        x = self.pos_encoder(x)

        # spin through the modules defined within the recurrent network portion of the model
        for m in self.encoder:
            x = m(x)
            if self.config['check_outputs']:
                output_checks[f'{m.__class__.__name__}-Layer{output_pos}'] = x; output_pos += 1

        # flatten the encoder output
        x = torch.flatten(x, start_dim=1) # squash down to 2d

        # finish up the pass regressing the sequence weights into class log odds
        for m in self.fc:
            x = m(x)
            if self.config['check_outputs']:
                output_checks[f'{m.__class__.__name__}-Layer{output_pos}'] = x; output_pos += 1

        return x, output_checks

    def reset_weights(self, mode='fold'):
        ''' Method for initial weights and resetting weights between cross-validation folds '''
        minit = ModelSupport(self.config)
        _ = minit.init_weights(self.embedding) if mode == 'init' else None
        minit.init_weights(self.encoder)
        minit.init_weights(self.fc)
        return

class PositionalEncoding(nn.Module):
    '''
        Positional embedding encoder

        Modified from Pytorch tutorial -> https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        if batch_first:
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x -> shape [seq_len, batch_size, embedding_dim] if not batch_first else [batch_size, seq_len, embedding_dim]
        """
        dim = 1 if self.batch_first else 0
        x = x + self.pe[:x.size(dim)]
        return self.dropout(x)


##############################################################################################################################
# Prebuilt Supervised Transformer Model
##############################################################################################################################

class SupervisedPrebuilt(nn.Module):
    ''' Prebuilt fine-tuning of transformer network for time-series model of email content 

        Using Roberta prebuilt model from Huggingface, api at https://huggingface.co/transformers/model_doc/roberta.html
    '''
    def __init__(self, config):
        super().__init__()
        self.number_classes = config['number_classes'] # number target classes
        self.dropout = config['dropout']
        self.encoder_layers = config['encoder_layers']
        self.config = config

        self.mconfig = RobertaConfig.from_pretrained(f'{self.config["pretrained_dir"]}{self.config["pretrained_model"]}')
        self.mconfig.update({"is_decoder":False, "num_layers":self.encoder_layers, "output_hidden_states":True, "hidden_dropout_prob": self.dropout, "layer_norm_eps": 1e-7})
        # self.model is created in reset_weights - if kfold CV is used then model will need to be reset multiple times
        
        # define network
        
        # Normalization layer
        ln_input_len = self.mconfig.to_dict()['hidden_size'] * 4
        self.norm = nn.ModuleList([
            nn.LayerNorm(ln_input_len),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        ])
        
        # Fully connected network to reduce encoder weights to log odds
        fc_input_len = self.mconfig.to_dict()['hidden_size'] * 4
        self.fc = nn.ModuleList([
            nn.Linear(fc_input_len, fc_input_len),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_input_len, fc_input_len // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_input_len // 2, self.number_classes),
            nn.LogSoftmax(),
        ])

        self.reset_weights() # in this case the prebuilt Roberta model is created in reset_weights

        self.to(DEVICE) # move to GPU if available
        return

    def forward(self, input_ids, attention_mask, **kwargs):
        ''' Forward pass for prebuilt network. '''
        output_checks = {}
        output_pos = 0

        layers = [-4, -3, -2, -1]

        outputs = self.model(input_ids, attention_mask)
        x = outputs.hidden_states
        amask = attention_mask.unsqueeze(-1).expand(x[layers[0]].size())

        x = torch.cat(tuple(torch.sum(x[l]*amask, dim=1) for l in layers), dim=1) # sum each of the last four layers by sequence then concatenate
        if self.config['check_outputs']:
            output_checks[f'{self.model.__class__.__name__}-Layer{output_pos}'] = x; output_pos += 1
            
        # process normalization layer
        for m in self.norm:
            x = m(x)
            if self.config['check_outputs']:
                output_checks[f'{m.__class__.__name__}-Layer{output_pos}'] = x; output_pos += 1        
            
        # finish up the pass regressing the sequence weights into class log odds
        for m in self.fc:
            x = m(x)
            if self.config['check_outputs']:
                output_checks[f'{m.__class__.__name__}-Layer{output_pos}'] = x; output_pos += 1

        return x, output_checks

    def reset_weights(self, mode=None):
        ''' Method for initial weights and resetting weights between cross-validation folds '''
        # Prebuilt network is instantiated here so if CV is used a fresh model is reintroduced for each new fold.
        self.model = RobertaModel.from_pretrained(f'{self.config["pretrained_dir"]}{self.config["pretrained_model"]}', config=self.mconfig).to(DEVICE)
        self.model.train()
        minit = ModelSupport(self.config)
        minit.init_weights(self.norm)
        minit.init_weights(self.fc)
        return


##############################################################################################################################
# Common model training and eval functions
##############################################################################################################################

class ModelManagement():
    ''' Common training/eval and state management routines for model creation. '''
    def __init__(self, model:nn.Module, config:dict, training_set:ContentDataset, eval_set:ContentDataset):
        self.config = config
        self.model = model
        self.training_set = training_set
        self.eval_set = eval_set

        # state variables
        self.prev_loss = sys.float_info.max
        self.prev_acc = 0.0

        # metrics
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

        # graphing functions
        self.graphing = ModelSupport(self.config)

    def training_plot(self):
        ''' Basic loss/accuracy performance graph - overwrites previous graph '''
        # setup plot framework
        fig, axes = plt.subplots(2, 1, figsize=(10,8))
        fig.subplots_adjust(top=0.90, bottom=0.1, wspace=0.35, hspace=0.80)
        axes = axes.flatten()
        # loss subplot
        ax = axes[0]
        ax.grid(True)
        ax.tick_params(axis='both', which='major')
        ax.set_xlim([-1, 50])
        ax.set_ylim([-0.01, 2.0])
        ax.set_title('Train Loss -vs- Test Loss', {'fontsize':12})
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.plot(self.train_loss,'-bo')
        ax.plot(self.test_loss,'-go')
        ax.legend(['Train Loss','Test Loss'])
        # accuracy subplot
        ax = axes[1]
        ax.grid(True)
        ax.tick_params(axis='both', which='major')
        ax.set_xlim([-1, 50])
        ax.set_ylim([0.2, 1.0])
        ax.set_title('Train Accuracy -vs- Test Accuracy', {'fontsize':12})
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.plot(self.train_acc,'-ro')
        ax.plot(self.test_acc,'-co')
        ax.legend(['Train Accuracy','Test Accuracy'])

        fig.savefig(f'{self.config["graph_dir"]}{"Metrics"}_{self.config["model_id"]}.png', facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    def state_management(self, fold, epoch, loss_train, loss_test=None, acc_train=None, acc_test=None, plot=True):
        ''' Save checkpoints and output performance graph. '''

        # metrics for performance graph
        self.train_loss.append(loss_train)
        self.test_loss.append(loss_test)
        self.train_acc.append(acc_train)
        self.test_acc.append(acc_test)

        # calculate progress
        working_loss = loss_train if loss_test is None else loss_test # use test loss for comparison if provided
        self.prev_loss = working_loss if working_loss < self.prev_loss else self.prev_loss
        working_acc = acc_train if acc_test is None else acc_test
        self.prev_acc = working_acc if working_acc > self.prev_acc else self.prev_acc

        if self.prev_loss == working_loss and self.prev_acc == working_acc and self.config['cv_mode'] == 'blend': # best model output so far, save it if blending CV
            fn = self.config['checkpoint_fn'].format(dir=self.config['checkpoint_dir'], fold='0', id=self.config['model_id']) # '{dir}fold_{fold}_{id}_checkpoint.pt',
            torch.save(self.model.state_dict(), fn)
        if epoch+1 == self.config['epochs'] and self.config['cv_mode'] != 'blend': # save each completed fold if using k-fold CV
            fn = self.config['checkpoint_fn'].format(dir=self.config['checkpoint_dir'], fold=str(fold), id=self.config['model_id']) # '{dir}fold_{fold}_{id}_checkpoint.pt',
            torch.save(self.model.state_dict(), fn)

        if plot:
            self.training_plot()

        if epoch+1 == self.config['epochs'] and self.config['spot_check']: # spot check weights, biases and gradients
            self.graphing.graph_parameters(self.model, fold, epoch, 0, types=['weight','bias'])
            self.graphing.graph_parameters(self.model, fold, epoch, 0, types=['grad']) 

        return # could add patience and divergence checks for early stopping

    def training(self):
        ''' Standard training loop w/ optional cross validation '''
        bs = self.config['batch_size']
        lr = self.config['learning_rate']
        epochs = self.config['epochs']

        dataset = self.training_set
        step_check = 5 * math.ceil(int((len(dataset) - len(dataset)*self.config['test_size']) // (bs*5)) / 5)

        kfolds = KFold(n_splits=self.config['kfolds'])
        
        loss_function = nn.NLLLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # spot check initial weights, biases if option set
        if self.config['spot_check']:
            self.graphing.graph_parameters(self.model, 0, 0, 0, types=['weight','bias'])

        print(f'\n--- Training model {self.config["model_id"]} in mode "{self.config["cv_mode"]}" with {1 if self.config["cv_mode"]=="blend" else self.config["kfolds"]} folds\n')

        # setup k-fold cross-validation 
        for fold, (train_idx, test_idx) in enumerate(kfolds.split(dataset)):
            train_loader = DataLoader(dataset, batch_size=bs, sampler=SubsetRandomSampler(train_idx), drop_last=True)
            test_loader = DataLoader(dataset, batch_size=bs, sampler=SubsetRandomSampler(test_idx), drop_last=True)

            # if using k-fold "properly", reset model weights between folds
            if self.config['cv_mode'] != 'blend' and fold > 0:
                self.model.reset_weights(mode='init')

            # epochs per fold
            for e in range(0, epochs):

                ########
                # train
                ########
                losses_t = []
                acc_t = []
                self.model.train()
                for step, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    batch = {k:v.to(DEVICE) for k,v in batch.items()}
                    predictions, outputs = self.model(**batch)

                    loss = loss_function(predictions, (torch.tensor)(batch['label']).squeeze())
                    loss.backward()
                    if self.config['clip_gradients']:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_max_norm']) # ensure exploding gradients are managed if not debugging

                    losses_t.append(loss.detach().cpu().item())
                    acc_t.append(torch.sum(torch.argmax(predictions, dim=1, keepdim=True) == batch['label']).detach().cpu().item())
                    if step % step_check == 0:
                        print('- Fold %2d / Epoch %2d / Step %3d --- train nll %.10f (acc %.10f)' % (fold, e, step, np.mean(losses_t), np.sum(acc_t)/((step+1)*bs)), flush=True)

                        if self.config['check_outputs']: # visualize output distribution
                            self.graphing.graph_outputs(fold, e, step, outputs)
                        if self.config['check_weights']: # visualize weight distribution
                            self.graphing.graph_parameters(self.model, fold, e, step, types=['weight','bias'], spot_check=False)
                        if self.config['check_gradients']: # visualize gradients or clip
                            self.graphing.graph_parameters(self.model, fold, e, step, types=['grad'], spot_check=False) 

                    optimizer.step()
                    del batch

                ########
                # test
                ########
                losses_e = []
                acc_e = []
                self.model.eval()
                for batch in test_loader:
                    batch = {k:v.to(DEVICE) for k,v in batch.items()}
                    with torch.no_grad():
                        predictions, _ = self.model(**batch)
                        loss = loss_function(predictions, (torch.tensor)(batch['label']).squeeze())
                        losses_e.append(loss.detach().cpu().item())
                        acc_e.append(torch.sum(torch.argmax(predictions, dim=1, keepdim=True) == batch['label']).detach().cpu().item())
                    del batch

                # calculate performance
                train_nll = np.mean(losses_t)
                test_nll = np.mean(losses_e)

                train_acc = np.sum(acc_t) / (bs * len(train_loader))
                test_acc = np.sum(acc_e) / (bs * len(test_loader))

                self.state_management(fold, e, loss_train=train_nll, loss_test=test_nll, acc_train=train_acc, acc_test=test_acc, plot=True)
                print('\n--- Fold %2d / Epoch %2d --- train nll %.10f (acc %.10f), --- test nll %.10f (acc %.10f)\n' % (fold, e, train_nll, train_acc, test_nll, test_acc), flush=True)
        return

    def evaluation(self):
        ''' 
            Method for evaluating a single model's effectiveness post training.
            Includes logic to aggregate an ensemble outcome if multiple fold checkpoints are available.
        '''
        bs = self.config['batch_size']

        dataset = self.eval_set

        eval_loader = DataLoader(dataset, batch_size=bs, drop_last=False)
        loss_function = nn.NLLLoss()

        # determine model iterations (i.e. using k-fold checkpoints or blended checkpoint)
        folds = 1 if self.config['cv_mode'] == 'blend' else self.config['kfolds']
        print(f'\n--- Evaluating model {self.config["model_id"]} in mode "{self.config["cv_mode"]}" with {1 if self.config["cv_mode"]=="blend" else self.config["kfolds"]} folds')

        ensemble_preds = []
        ensemble_labels = []
        for f in range(folds):

            model_fn = self.config['checkpoint_fn'].format(dir=self.config['checkpoint_dir'], fold=str(f), id=self.config['model_id'])
            if not os.path.exists(model_fn):
                raise AssertionError('model checkpoint does not exist...something is wrong')
            self.model.load_state_dict(torch.load(model_fn, map_location=DEVICE))

            ########
            # eval
            ########
            preds_e = []
            labels_e = []
            losses_e = []
            acc_e = []
            self.model.eval()
            for batch in eval_loader:
                batch = {k:v.to(DEVICE) for k,v in batch.items()}
                with torch.no_grad():
                    predictions, _ = self.model(**batch)
                    loss = loss_function(predictions, (torch.tensor)(batch['label']).squeeze())

                    # collect metrics
                    losses_e.append(loss.detach().cpu().item())
                    acc_e.append(torch.sum(torch.argmax(predictions, dim=1, keepdim=True) == batch['label']).detach().cpu().item())
                    preds_e.extend(predictions.detach().cpu().tolist())
                    labels_e.extend(batch['label'].detach().cpu().tolist())
                del batch

            # save fold predictions for ensemble calculations
            ensemble_preds.append(preds_e)
            ensemble_labels = np.array(labels_e).squeeze(axis=1) if len(ensemble_labels) == 0 else ensemble_labels

            # calculate metrics
            eval_nll = np.mean(losses_e)
            eval_acc = np.sum(acc_e) / (len(dataset))

            print('\n--- Model "%s" fold "%d" evaluation nll loss of %.10f with %.10f accuracy' % (self.config['model_id'], f, eval_nll, eval_acc))

        # ensemble calculations
        ensemble_preds = np.transpose(ensemble_preds, (1, 0, 2)) # alter matrix to (samples X folds X prediction probabilities)
        ensemble_preds = np.sum(ensemble_preds, axis=1) # sum all the probabilities by class and fold
        ensemble_preds = np.argmax(ensemble_preds, axis=1) # select the class with the highest sum
        acc_s = np.sum(ensemble_preds == ensemble_labels) / len(dataset) # compare predictions with actuals

        print('\n--- Ensemble "%s" accuracy prediction is %.10f' % (self.config['model_id'], acc_s))

        return


##############################################################################################################################
# Configuration
##############################################################################################################################
class PipelineConfig():
    ''' Common configuration class for Training and Inference pipeline logic. '''
    config = { # common configuration variables
        'data_dir': '{base}/data/',
        'checkpoint_dir': '{base}/checkpoints/{cv}/',
        'graph_dir': '{base}/graphs/',
        'pretrained_dir':'{base}/',
        'pretrained_model':'roberta-base',
        'supervised_dataset_fn': 'supervised_email_train.csv',
        'vocabulary_fn': 'supervised_email.vocab',
        'checkpoint_fn':'{dir}fold_{fold}_{id}_checkpoint.pt',
        # common control variables
        'test_size': 0.1, # split of training data for eval
        'limit': 1.0, # restrict input data for debugging performance
        'check_outputs': False,  # flag to graph/analysis layer outputs
        'check_weights': False,  # flag to graph/analysis weights/biases
        'check_gradients': False,  # flag to graph/analysis gradients
        'spot_check': True, # flag to graph parameters/gradients after every epoch
        'cv_mode': 'fold', # use proper k-fold CV or blend into one checkpoint
        'force_train': False, # force rebuild
    }

    # model specific configuration
    model_config = {
        'lstm': {'kfolds': 3, 'epochs': 15, 'batch_size': 128, 'embedding_len': 128, 'embedding_type':'train',
                 'max_tokens': 192, 'dropout': 0.6, 'learning_rate':8e-05, 'bidirectional':True, 'rnn_layers':2, 
                 'clip_gradients':False, 'clip_max_norm':5.0},
        'gru': {'kfolds': 3, 'epochs': 15, 'batch_size': 128, 'embedding_len': 128, 'embedding_type':'train',
                'max_tokens': 192, 'dropout': 0.7, 'learning_rate':8e-05, 'bidirectional':True, 'rnn_layers':2,
                'clip_gradients':False, 'clip_max_norm': 5.0},
        'trn': {'kfolds': 3, 'epochs': 20, 'batch_size': 64, 'embedding_len': 128, 'embedding_type':'train',
                'max_tokens': 192, 'dropout': 0.2, 'learning_rate':8e-05, 'attention_heads':8, 'encoder_layers':4,
                'clip_gradients':False, 'clip_max_norm': 5.0}, 
        'pre': {'kfolds': 3, 'epochs': 5, 'batch_size': 64, 'embedding_type':'hgf_pretrained',
                'max_tokens': 192, 'dropout': 0.1, 'learning_rate':1e-04, 'encoder_layers':1,
                'clip_gradients':False, 'clip_max_norm': 5.0}, 
    }

    def __init__(self, config):
        # tack on custom config if present
        _ = self.config.update(config) if config is not None else None
        # set directory structure and create if needed
        if IS_LOCAL:
            self.config.update({'data_dir':str(self.config['data_dir']).format(base=f'{LOCAL_BASEDIR}')})
            self.config.update({'checkpoint_dir':str(self.config['checkpoint_dir']).format(base=f'{LOCAL_BASEDIR}', cv=self.config['cv_mode'])})
            self.config.update({'graph_dir':str(self.config['graph_dir']).format(base=f'{LOCAL_BASEDIR}')})
            self.config.update({'pretrained_dir':str(self.config['pretrained_dir']).format(base=f'{LOCAL_PRETRAIN_BASEDIR}')})
        else: # running in Kaggle Environment
            self.config.update({'data_dir':str(self.config['data_dir']).format(base=f'{KAGGLE_BASEDIR}/input/emailblog')})
            self.config.update({'checkpoint_dir':str(self.config['checkpoint_dir']).format(base=f'{KAGGLE_BASEDIR}/working', cv=self.config['cv_mode'])})
            self.config.update({'graph_dir':str(self.config['graph_dir']).format(base=f'{KAGGLE_BASEDIR}/working')})
            self.config.update({'pretrained_dir':str(self.config['pretrained_dir']).format(base=f'{KAGGLE_BASEDIR}/input/robertamodels')})
        if not os.path.exists(self.config['checkpoint_dir']):
            os.makedirs(self.config['checkpoint_dir'], mode=711, exist_ok=True)
        if not os.path.exists(self.config['graph_dir']):
            os.makedirs(self.config['graph_dir'], mode=711, exist_ok=True)
        return

    def get_config(self, model=None) -> dict:
        ''' Returns the current base configuration plus model configuration if requested. '''
        config = self.config
        _ = config.update(self.model_config[model]) if model is not None else None
        return config


##############################################################################################################################
# Training/Test Pipeline
##############################################################################################################################

class TrainingPipeline():
    ''' Basic routine to control data prep and modeling flow for each specific model and type requested.'''

    def __init__(self, id:str, label_column:str, label_classes:int, config:dict=None, run_models=['lstm','gru','trn','pre']):
        self.id = id
        self.run_models = run_models
        self.label_column = label_column
        self.label_classes = label_classes

        self.config = PipelineConfig(config).get_config().copy()
        self.config.update({'input_columns':{'Body':'content', self.label_column:'label'}, 'number_classes': self.label_classes})

        # load training data
        raw_data = RawDataLoader(self.config) # prep input data
        self.train_df, self.eval_df = raw_data.split_dataset(split=self.config['test_size'])
        self.vocab = raw_data.vocab
        self.config.update({'vocab_size':len(self.vocab)})
        return

    def run_pipeline(self):
        ''' Main train/eval processing loop. '''

        # process all of the models in run_models
        for m in self.run_models:
            start = time()
            config = PipelineConfig(None).get_config(model=m).copy()
            config.update({'model_id':f'{m}-{self.id}'})
            config.update({'input_columns':{'Body':'content', self.label_column:'label'}, 'number_classes': self.label_classes})
            config.update({'vocab_size':len(self.vocab)})

            pipeline = self._get_pipeline(m, config)
            
            # skip train if exists already and train_mode is 'normal'
            _ = pipeline.training() if config['force_train'] or not self.is_training_complete(config['model_id']) else None
            # evaluate model
            pipeline.evaluation()
            
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()

            print(f'\n--- Pipeline runtime for model_id "{config["model_id"]}" complete in {time()-start} seconds\n')
        return

    def is_training_complete(self, model_id):
        ''' Determine if model checkpoint already exists. '''
        filter = self.config['checkpoint_fn'].format(dir=self.config['checkpoint_dir'], fold=0, id=model_id)
        checkpoints = glob.glob(filter)
        return len(checkpoints) > 0

    def _get_pipeline(self, id, config):
        ''' Common function for setting up model training/eval pipeline '''
        model = SupervisedRNN(id, config) if id in ['lstm','gru'] else SupervisedTransformer(config) if id=='trn' else SupervisedPrebuilt(config)
        training_set = ContentDataset(self.train_df, config, vocab=self.vocab)
        eval_set = ContentDataset(self.eval_df, config, vocab=self.vocab)
        pipeline = ModelManagement(model, config, training_set, eval_set)
        return pipeline


##############################################################################################################################
# Inference Pipeline
##############################################################################################################################

class InferencePipeline(): 
    '''Inference pipeline logic. '''

    # some fictitious emails
    samples = [
        'Hey! We\'re planning on visiting the lake house this weekend, do you want to go along?  -Robert',
        'JoAnn, the TPS report you promised to have yesterday is not on my desk. Please have that to me by the end of the day.',
        'Attached is the NYOs listing notice for training in Topeka next week. If you would like to attend, make arrangements ASAP as space is limited.',
        'Good Morning John, hope you\'re doing well.  It has been very rainy here.  Anyway, I wanted to ask you a question on the odds of me being picked for early retirement? Let me know. Steve',
        'Hi, thanks for the email and keeping me in the loop.  My daughter\'s classes at school have been hectic and I need a break this weekend.  The open house is this Saturday, hopefully I will be able to see you there. Sue',
        'Thanks for the heads up!',
        'There\'s a business conference next week and I think there are several good bars in the area.  We should plan to meet up there for drinks afterwards.',
        'Please plan to attend the quarterly disaster preparation meeting this Tuesday.  The location is TBD, but the time will from 10a - 2p.  Lunch will be included.',
        'Dave, good time playing poker last week.  I\'m heading out for a round of golf this afternoon and could use a partner.  How about 2pm?',
        'The weather today is expected to be rainy with a chance of thunderstorms and then clearing off for tomorrow...',
        'The systems here are awful!  The building is rundown and in shambles. I wish they would fix this mess instead of sucking up to investors.  I hate this!',
        'I expect many arrests soon given the catastrophic consequences.',
        'I\'m bored and I hate my job.  My life is terrible, unfair and full of regrets.  My coworkers are being jerks and are the worst kind of people.',
    ]

    def __init__(self, run_labels:dict, label_classes:int, config:dict=None, run_models=['lstm','gru','trn','pre']):
        self.run_models = run_models
        self.run_labels = run_labels
        self.label_classes = label_classes

        self.config = PipelineConfig(config).get_config().copy()
        self.vocab = Vocabulary(self.config).get_vocabulary()

        return

    def run_pipeline(self, samples=None):
        ''' 
            Inference prediction processing, runs samples through each model in run_models and for each label in run_labels.
            Aggregates scores and prints final results.
        '''
        start = time()

        # build data loader for inference samples
        self.samples = samples if samples is not None else self.samples
        outputs = pd.DataFrame(self.samples, columns=['emails'])
        inputs = pd.DataFrame(self.samples, columns=['content'])

        # process each model within each label type
        ensemble_preds = []
        ensemble_votes = []
        model_count = 0
        for key in self.run_labels.keys(): # e.g. cs1, cs2, cs3
            for m in self.run_models: # e.g. lstm, gru, trn, pre
                config = PipelineConfig(None).get_config(model=m).copy()
                config.update({'number_classes': self.label_classes})
                config.update({'model_id':f'{m}-{key}'})
                config.update({'vocab_size':len(self.vocab)})

                model_fn_wildcard = config['checkpoint_fn'].format(dir=config['checkpoint_dir'], fold='*', id=config['model_id'])
                model_checkpoint_fns = glob.glob(model_fn_wildcard)
                if len(model_checkpoint_fns) <= 0:
                    print(f'\n!!! Error - no checkpoints for model {config["model_id"]}')
                    continue

                print(f'\n--- Infering model {config["model_id"]} in "{config["cv_mode"]}" mode with {len(model_checkpoint_fns)} folds')
                model_count += 1

                inference_set = ContentDataset(inputs, config, vocab=self.vocab)
                inference_loader = DataLoader(inference_set, batch_size=config['batch_size'], drop_last=False)
                model = SupervisedRNN(m, config) if m in ['lstm','gru'] else SupervisedTransformer(config) if m=='trn' else SupervisedPrebuilt(config)

                for model_fn in model_checkpoint_fns:

                    # load the model state for the current fold checkpoint
                    model.load_state_dict(torch.load(model_fn, map_location=DEVICE))

                    ########
                    # infer
                    ########
                    preds_e = []
                    model.eval()
                    for batch in inference_loader:
                        batch = {k:v.to(DEVICE) for k,v in batch.items()}
                        with torch.no_grad():
                            predictions, _ = model(**batch)
                            preds_e.extend(predictions.detach().cpu().tolist())
                        del batch

                    # save fold predictions for ensemble calculations
                    ensemble_preds.append(preds_e)
                    ensemble_votes.append(np.argmax(preds_e, axis=1).tolist())

                del model
                gc.collect()
                torch.cuda.empty_cache()
            
        # ensemble predictions
        ensemble_preds = np.transpose(ensemble_preds, (1, 0, 2)) # alter matrix to (samples X folds X prediction probabilities)
        ensemble_preds = np.sum(ensemble_preds, axis=1) # sum all the probabilities by class and fold
        ensemble_preds = np.argmax(ensemble_preds, axis=1) # select the class with the highest sum
        outputs['pred_prob'] = ensemble_preds.tolist()

        ensemble_votes = np.transpose(ensemble_votes, (1, 0))
        ensemble_votes = np.median(ensemble_votes, axis=1).astype(int)
        outputs['pred_vote'] = ensemble_votes.tolist()

        print(f'\n--- Inference pipeline runtime for {model_count} models complete in {time()-start} seconds\n')
        print(outputs.head(50))

        return outputs


##############################################################################################################################
# Main - Train/Eval Processing
##############################################################################################################################
print(f'\n--- Device is {DEVICE}')

#
# Input dataframe feature columns: 
#  From_Address, To_Address, CC_Address
#  DateTime, DateTime_HOUR, DateTime_MONTH, DateTime_TS, Day, Outside_Hours
#  Body, Subject, Source, Forwarded
#
# labels = ['Class_Sentiment_1','Class_Sentiment_2','Class_Sentiment_Vader','Class_Alignment_1','Class_Alignment_2','Class_Alignment_3']
# models = ['lstm','gru','trn','pre']
#
run_labels = {'cs1':'Class_Sentiment_1','cs2':'Class_Sentiment_2','csv':'Class_Sentiment_Vader'}
run_models = ['lstm','gru','trn'] # 'pre'

# Process each label type and network model defined in 'run_labels' and 'run_models'
# 'blend' will not reset weights between folds, use some other value (i.e. not 'blend') to separate folds for mean selection
for key, run_label in run_labels.items():
    pipeline = TrainingPipeline(key, run_label, 3, config=None, run_models=run_models)
    pipeline.run_pipeline()


##############################################################################################################################
# Main - Inference Testing
##############################################################################################################################

# Aggregate models to predict most likely outcome
# TODO add weights
run_labels = {'cs1':''} 
run_models = ['trn']
predictions = InferencePipeline(run_labels, 3, run_models=run_models).run_pipeline()

exit()