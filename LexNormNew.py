#!/usr/bin/env python
# coding: utf-8

# # Lexical normalization pipeline 
# 
# author - AR Dirkson 
# date - 15-7-2019
# 
# Python 3 script
# 
# This pipeline takes raw text data and performs: 
# - Removes URLs, email addresses
# - Tokenization with NLTK
# - Removes non_English posts (conservatively) using langid module with top 10 languages and threshold of 100
# - British English to American English 
# - Normalization of contractions
# - Normalization of generic abbreviations and slang 
# - Normalization of domain-specific (patient forum) abbreviations 
# - Spelling correction 

# In[1]:


import pickle
import numpy as np
import random
import pandas as pd
from collections import Counter, defaultdict, OrderedDict
from nltk import pos_tag, word_tokenize
import re
import seaborn as sns
import matplotlib.pyplot as plt
import editdistance

import kenlm
from sklearn.metrics import recall_score, precision_score, f1_score, fbeta_score
from nltk.tokenize.treebank import TreebankWordDetokenizer 
from gensim.models import KeyedVectors


# In[2]:


class Normalizer (): 
        
    def __init__(self): 
        pass
        
    #to use this function the files need to be sorted in the same folder as the script under /obj_lex/
    def load_obj(self, name):
        with open('obj_lex\\' + name + '.pkl', 'rb') as f:
            return pickle.load(f, encoding='latin1')
        
    def load_files(self): 
        self.abbr_dict = self.load_obj ('abbreviations_dict')
        self.aspell_dict = self.load_obj ('aspell_dict_lower')       
        self.short_expanse_dict = self.load_obj ('short_expansions_dict')
        self.cList = self.load_obj ('contractionslistone')
        self.cList2 = self.load_obj ('contractionslisttwo')
        self.drugnames = self.load_obj ('fdadrugslist')

    def change_tup_to_list(self, tup): 
        thelist = list(tup)
        return thelist
    
    def change_list_to_tup(self,thelist): 
        tup = tuple(thelist)
        return tup
    
#---------Remove URls, email addresses and personal pronouns ------------------
        
    def replace_urls(self,list_of_msgs): 
        list_of_msgs2 = []
        for msg in list_of_msgs: 
            nw_msg = re.sub(
        r'\b' + r'((\(<{0,1}https|\(<{0,1}http|\[<{0,1}https|\[<{0,1}http|<{0,1}https|<{0,1}http)(:|;| |: )\/\/|www.)[\w\.\/#\?\=\+\;\,\&\%_\n-]+(\.[a-z]{2,4}\]{0,1}\){0,1}|\.html\]{0,1}\){0,1}|\/[\w\.\?\=#\+\;\,\&\%_-]+|[\w\/\.\?\=#\+\;\,\&\%_-]+|[0-9]+#m[0-9]+)+(\n|\b|\s|\/|\]|\)|>)',
        '-URL-', msg)
            list_of_msgs2.append(nw_msg)
        return list_of_msgs2    

    def replace_email(self,list_of_msgs): 
        list_of_msgs2 = []
        for msg in list_of_msgs: 
            nw_msg = re.sub (r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+[. ])", ' ', msg) #remove email
            nw_msg2 = re.sub (r"(@[a-zA-Z0-9]+[. ])", ' ', nw_msg) #remove usernames
#             nw_msg3 = re.sub(r"(@ [a-zA-Z0-9]+[. ])", ' ', nw_msg2) #remove usernames
            list_of_msgs2.append(nw_msg2)
        return list_of_msgs2

    def remove_empty (self,list_of_msgs): 
        empty = []
        check_msgs3 =[]
        for a, i in enumerate (list_of_msgs): 
            if len(i) == 0: 
                print('empty')
            else: 
                check_msgs3.append(i)
        return check_msgs3

        
    def remove_registered_icon (self, msg): 
        nw_msg = re.sub ('\u00AE', '', msg)
        nw_msg2 = re.sub ('\u00E9', 'e', nw_msg)
        return nw_msg2
      
    def escape_char (self, msg): 
        msg1 = msg.replace('\x08', '') 
        msg2 = msg1.replace ('\x8d', '')
        msg3 = msg2.replace('ðŸ', '')
        return msg3

    def anonymize (self, posts): 
        posts2 = self.replace_urls (posts)
        posts3 = self.replace_email (posts2)
        posts4 = self.remove_empty(posts3)
        posts5 = [self.remove_registered_icon(p) for p in posts4]
        posts6 = [self.escape_char(p) for p in posts5]
        return posts6

#---------Convert to lowercase ----------------------------------------------------
    
    def lowercase (self, post):
        post1 = []
        for word in post: 
            word1 = word.lower()
            post1.append (word1)
        return post1

#---------Remove non_English posts -------------------------------------------------    
    def language_identify_basic (self, posts):
        nw = []
        tally = 0
        list_removed = []
        for post in posts: 
            out = langid.classify (post)
            out2 = list(out)
            if out2[0]=='en': 
                nw.append(post)
            else: 
                tally += 1 
                list_removed.append(tuple ([post, out2[0], out2[1]]))
        return nw, tally, list_removed
    
    def language_identify_thres (self, msgs, lang_list, thres):
        nw = []
        tally = 0
        list_removed = []
        for post in msgs: 
            langid.set_languages(lang_list)
            out = langid.classify (post)
            out2 = list(out)
            if out2[0]=='en': 
                nw.append(post)
            elif out2[1] > thres:
                nw.append(post)
            else: 
                tally += 1 
                list_removed.append(tuple ([post, out2[0], out2[1]]))
        return nw, tally, list_removed   

    
    def remove_non_english(self, posts): 
#         d = TreebankWordDetokenizer()
#         posts2 = [d.detokenize(m) for m in posts]
        
        posts_temp, tally, list_removed = self.language_identify_basic(posts)        
        lang = []

        for itm in list_removed: 
            lang.append(itm[1])

        c = Counter(lang)

        lang_list = ['en']

        for itm in c.most_common(10): 
            z = list(itm)
            lang_list.append(z[0])
    
        print("Most common 10 languages in the data are:" + str(lang_list))
        posts3, tally_nw, list_removed_nw = self.language_identify_thres(posts, lang_list, thres = -100)
        return posts3

## --- Contraction expansions ------------------------------##

    def prepareContractions(self):
        self.c_re = re.compile('(%s)' % '|'.join(self.cList.keys()))
        self.c_re2 = re.compile('(%s)' % '|'.join(self.cList2.keys()))
        
    def remove_apos (self, sent): 
        sent2 = re.sub ("'",'', sent)
        return sent2
#         except TypeError: 
#             pass

    def expandContractions (self, text):
        def replace(match):
            return self.cList[match.group(0)]
        return self.c_re.sub(replace, text)
    
    #needs to happen after tokenization
    def expandContractions_second (self, text):
        text2 = []
        for w in text:
            if w.lower() in self.cList2: 
                v = word_tokenize(self.cList2[w.lower()])
                for i in v: 
                    text2.append(i)
            else:
                text2.append(w) 
        return text2   

###--- 1-2 letter expansions -------------------------------## 
    def load_ngrammodel(self): 
        path = 'obj_lex\\tetragram_model.binary'
        self.model = kenlm.Model(path)
    
    def get_parameters_ngram_model (self, word, sent): 
        i = sent.index(word)
        if ((i-2) >= 0) and (len(sent)>(i+2)):
            out = sent[(i-2):(i+3)]
            bos = False
            eos = False
        elif ((i-2) < 0) and (len(sent)> (i+2)) :  #problem with beginning
            bos = True
            eos = False
            out = sent[0:(i+3)]
        elif ((i-2) >= 0) and (len(sent) <= (i+2)): #problem with end
            bos = False
            eos = True
            out = sent[(i-2):]
        else: #problem with both
            out = sent
            eos = True
            bos = True  
        d = TreebankWordDetokenizer()
        out2 = d.detokenize(out)            
        return bos, eos, out2
    
    def get_prob(self, word, token, out, bos, eos): #token is candidate
        out_nw = out.replace(word, token)
        p = self.model.score(out_nw, bos = bos, eos = eos)
        return p


    def short_abbr_expansion(self, sent): 
        sent2 = []
        for word in sent: 
            if len(word) > 2: 
                sent2.append(word)
            else: 
                if word in self.short_expanse_dict .keys(): 
                    cand = self.short_expanse_dict [word]
                    final_p = -100
                    bos, eos, out = self.get_parameters_ngram_model(word,sent)
                    for i in cand: 
                        p = self.get_prob(word, i, out, bos, eos) 
                        if p > final_p: 
                            final_p = p 
                            correct = i
                    sent2.append(correct)
                else: 
                    sent2.append(word)
        return sent2
    
#---------Lexical normalization pipeline (Sarker, 2017) -------------------------------

    def loadItems(self):
        '''
        This is the primary load function.. calls other loader functions as required..
        '''    
        global english_to_american
        global noslang_dict
        global IGNORE_LIST_TRAIN
        global IGNORE_LIST

        english_to_american = {}
        lexnorm_oovs = []
        IGNORE_LIST_TRAIN = []
        IGNORE_LIST = []

        english_to_american = self.loadEnglishToAmericanDict()
        noslang_dict = self.loadDictionaryData()
        for key, value in noslang_dict.items (): 
            value2 = value.lower ()
            value3 = word_tokenize (value2)
            noslang_dict[key] = value3

        return None


    def loadEnglishToAmericanDict(self):
        etoa = {}

        english = open('obj_lex/englishspellings.txt')
        american = open('obj_lex/americanspellings.txt')
        for line in english:
            etoa[line.strip()] = american.readline().strip()
        return etoa

    def loadDictionaryData(self):
        '''
        this function loads the various dictionaries which can be used for mapping from oov to iv
        '''
        n_dict = {}
        infile = open('obj_lex/noslang_mod.txt')
        for line in infile:
            items = line.split(' - ')
            if len(items[0]) > 0 and len(items) > 1:
                n_dict[items[0].strip()] = items[1].strip()
        return n_dict



    def preprocessText(self, tokens, IGNORE_LIST, ignore_username=False, ignore_hashtag=False, ignore_repeated_chars=True, eng_to_am=True, ignore_urls=False):
        '''
        Note the reason it ignores hashtags, @ etc. is because there is a preprocessing technique that is 
            designed to remove them 
        '''
        normalized_tokens =[]
        #print tokens
        text_string = ''
        # NOTE: if nesting if/else statements, be careful about execution sequence...
        
#         tokens2 = [t[0] for t in tokens]
        
        for t in tokens:
            t_lower = t.strip().lower()
            # if the token is not in the IGNORE_LIST, do various transformations (e.g., ignore usernames and hashtags, english to american conversion
            # and others..
            if t_lower not in IGNORE_LIST:
                # ignore usernames '@'
                if re.match('@', t) and ignore_username:
                    IGNORE_LIST.append(t_lower)
                    text_string += t_lower + ' '
                #ignore hashtags
                elif re.match('#', t_lower) and ignore_hashtag:
                    IGNORE_LIST.append(t_lower)
                    text_string += t_lower + ' '
                #convert english spelling to american spelling
                elif t.strip().lower() in english_to_american.keys() and eng_to_am:    
                    text_string += english_to_american[t.strip().lower()] + ' '
                #URLS
                elif re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', t_lower) and ignore_urls:
                    IGNORE_LIST.append(t_lower)
                    text_string += t_lower + ' '                
                elif not ignore_repeated_chars and not re.search(r'[^a-zA-Z]', t_lower):
                    # if t_lower only contains alphabetic characters
                    t_lower = re.sub(r'([a-z])\1+', r'\1\1', t_lower)
                    text_string += t_lower + ' '  
                    # print t_lower

                # if none of the conditions match, just add the token without any changes..
                else:
                    text_string += t + ' '
            else:  # i.e., if the token is in the ignorelist..
                text_string += t_lower + ' '
            normalized_tokens = text_string.split()
        return normalized_tokens, IGNORE_LIST


    def dictionaryBasedNormalization(self, tokens, I_LIST, M_LIST):
        tokens2 =[]
        for t in (tokens):
            t_lower = t.strip().lower()
            if t_lower in noslang_dict.keys() and len(t_lower)>2:
                nt = noslang_dict[t_lower]
                [tokens2.append(m) for m in nt]

                if not t_lower in M_LIST:
                    M_LIST.append(t_lower)
                if not nt in M_LIST:
                    M_LIST.append(nt)
            else: 
                tokens2.append (t)
        return tokens2, I_LIST, M_LIST
    
#----Using the Sarker normalization functions ----------------------------
#Step 1 is the English normalization and step 2 is the abbreviation normalization

    def normalize_step1(self, tokens, oovoutfile=None):
        global IGNORE_LIST
        global il
        MOD_LIST = []
        # Step 1: preprocess the text
        normalized_tokens, il = self.preprocessText(tokens, IGNORE_LIST)
        return normalized_tokens
    
    def normalize_step2(self, normalized_tokens, oovoutfile=None): 
        global IGNORE_LIST
        global il
        MOD_LIST = []    
        ml = MOD_LIST
        normalized_tokens, il, ml = self.dictionaryBasedNormalization(normalized_tokens, il, ml)
        return normalized_tokens

    def sarker_normalize (self,list_of_msgs): 
        self.loadItems()
        msgs_normalized = [self.normalize_step1(m) for m in list_of_msgs]
        msgs_normalized2 = [self.normalize_step2(m) for m in msgs_normalized]    
        return msgs_normalized2

#-------Domain specific abreviation expansion ----------------------------
# The list of abbreviations is input as a dictionary with tokenized output  

    def domain_specific_abbr (self, tokens, abbr): 
        post2 = [] 
        for t in tokens:
            if t.lower() in abbr.keys(): 
                nt = abbr[t.lower()]
                [post2.append(m) for m in nt]
            else: 
                post2.append(t)
        return post2

    def expand_abbr (self, data, abbr): 
        data2 = []
        for post in data: 
            post2 = self.domain_specific_abbr (tokens = post, abbr= abbr)
            data2.append(post2)
        return data2
    
#-------Spelling correction -------------------------------------------------    
    def flev_rel (self, cand, token): 
        abs_edit_dist = editdistance.eval(cand, token)
        rel_edit_dist = abs_edit_dist / len(token)
        return rel_edit_dist
    
    def modelsim (self,cand,token, model): 
        try: 
            similarity = model.similarity(cand, token)
        except KeyError: 
            similarity = 0
        return similarity

    def run_low_emb (self, word, voc, model, w1 =0.4, w2= 0.6): 
        replacement = [' ',100]
        for token in voc: 
            sim1 = self.flev_rel(word, token) #lower is better
            sim2 = self.modelsim (word, token, model)
            sim = w1 * sim1 + w2 * (1-sim2)
            if sim < replacement[1]:
                replacement[1] = sim
                replacement[0] = token
        return replacement
    
    
    def wrong_concatenation(self, token, token_freq): 
        best_plausibility = 0
        best_split = 0
        t = token_freq[token]
        limit = 9*t
        NUMBER = re.compile('[0-9]+')  # Only letters and dashes

        if '-' in token: 
            return token
        else: 
            for i in range(3, len(token)):
                left, right = token[:i], token[i:]
                if len(right) < 3: 
                    continue

                elif NUMBER.fullmatch(left) and right in token_freq:
                    best_split= (left,right)

                elif NUMBER.fullmatch(right) and left in token_freq: 
                    best_split = (left, right)
                else:
                    if left not in token_freq or right not in token_freq: 
                        continue
                    if token_freq[left] < limit or token_freq[right] < limit: 
    #                     print('too low')
                        continue
                    plausibility = min(token_freq[left], token_freq[right])

                    if plausibility > best_plausibility:
                        best_plausibility = plausibility
                        best_split = (left, right)
            if best_split != 0:
                return list(best_split)
            else: 
                return token
    
    
    def spelling_correction (self, post, min_rel_freq = 9, max_flev_rel = 0.76): 
        post2 = []
        cnt = 0 
        tagged_post = pos_tag(post)
        tags = [t[1] for t in tagged_post]
        
        for a, token in enumerate (post): 
            token2 = token.lower()
            if (tags[a] == 'NNP') or (tags[a] == 'NNPS'): 
                post2.append(token)
            else:            
                if self.TRUE_WORD.fullmatch(token2) and (token2 != '-url-') and (token2 != '-') and (token2 != '--'):
#                     if token2 in self.spelling_corrections:
#                         correct = self.spelling_corrections[token2] 
#                         if len(correct) >1: 
#                             [post2.append(i) for i in correct]
#                         else: 
#                             post2.append(correct)
#                         cnt +=1
#                         self.replaced.append(token2)
#                         self.replaced_with.append(correct)

                    if token2 in self.aspell_dict:
                        post2.append(token)
            
                    elif token2 in self.drugnames: 
                        post2.append(token)

                    else:                   
                        freq_word = self.token_freq[token2]
                        limit = freq_word * min_rel_freq

                        subset = [t[0] for t in self.token_freq_ordered2 if t[1]>= limit] 
                        candidate = self.run_low_emb(token2, subset, self.model2)  

                        if candidate[1] > max_flev_rel: 
                            x = self.wrong_concatenation(token2, self.token_freq)
                            if x != token2:   
                                [post2.append(i) for i in x] 
                                cnt +=1
                                self.replaced.append(token2)
                                self.replaced_with.append( " ".join(x))
                                self.spelling_corrections [token2] = x
                            else: 
                                post2.append(token)
                        else: 
                            post2.append(candidate[0])
                            cnt +=1
                            self.replaced.append(token2)
                            self.replaced_with.append(candidate[0])
                            self.spelling_corrections [token2] = candidate[0]
                else: 
                    post2.append(token)
        
        self.total_cnt.append (cnt)
        return post2       
      
    def initialize_files_for_spelling(self): 
        total_cnt = []
        replaced = []
        replaced_with = []
        spelling_corrections= {}
        return total_cnt, replaced, replaced_with, spelling_corrections
    
    def change_tup_to_list (self, tup): 
        thelist = list(tup)
        return thelist

    
    def load_model (self): 
        filename = 'file://C://Users//dirksonar//Documents//Data//Stored_data//Corpora_from_others//HealthVec//Health_2.5mreviews.s200.w10.n5.v15.cbow.bin'
        #'file:///data/dirksonar/Project1_lexnorm/int_val/Health_2.5mreviews.s200.w10.n5.v15.cbow.bin'
        self.model2 = KeyedVectors.load_word2vec_format(filename, binary=True)

    def create_token_freq (self, data): 
        flat_data = [item for sublist in data for item in sublist]
        flat_data2 = [i.lower() for i in flat_data]
        flat_data3 = []
        for token2 in flat_data2: 
            if self.TRUE_WORD.fullmatch(token2) and (token2 != '-url-') and (token2 != '-') and (token2 != '--'):
                flat_data3.append(token2)
            
        self.token_freq = Counter(flat_data3)
        
        token_freq_ordered = self.token_freq.most_common ()
        self.token_freq_ordered2 = [self.change_tup_to_list(m) for m in token_freq_ordered]
    
    def correct_spelling_mistakes(self, data, different_token_freq = False):   
        self.load_model()
        self.load_files ()
        self.total_cnt, self.replaced, self.replaced_with, self.spelling_corrections = self.initialize_files_for_spelling()
        self.TRUE_WORD = re.compile('[-a-z]+')  # Only letters and dashes  
        if different_token_freq == False: 
            self.create_token_freq(data)
        else: 
            self.token_freq = self.load_obj('token_freq')
            token_freq_ordered = self.token_freq.most_common ()
            self.token_freq_ordered2 = [self.change_tup_to_list(m) for m in token_freq_ordered]
            
        out = []
        for num, m in enumerate(data): 
            if num%1000 == 0: 
                print(num)
            out.append(self.spelling_correction (m))
        return out, self.total_cnt, self.replaced, self.replaced_with, self.spelling_corrections    
    
#--------Overall normalization function--------------------------------------

    def normalize (self, posts):
        self.load_files ()
        posts0 = [str(m) for m in posts]
        posts1 = self.anonymize(posts0)
        print(posts1[0])
        posts2 = [i.replace ('’', "'") for i in posts1]
        self.prepareContractions()
        posts3 = [self.expandContractions(m) for m in posts2]

        posts4 = [self.remove_apos(m) for m in posts3]
       
        posts5 = [word_tokenize(m) for m in posts4]
        print('done with tokenizing')
        print(posts5[0])
        
        posts6 = [self.expandContractions_second(m) for m in posts5]
        print(posts6[0])

        self.load_ngrammodel()
        posts8 = [self.sarker_normalize(posts6)]
        posts8b = posts8[0]
        posts9 = [self.short_abbr_expansion(m) for m in posts8b]
        posts10 = [self.expand_abbr(posts9, self.abbr_dict)]   
        print(posts10[0][0])
        return posts10[0]
    


# In[4]:


#example of usage 

test = ['my bff is 4ever', 'the colour of the moon is grey']

#normalize but not correct spelling mistakes
test2 = Normalizer().normalize(test)

#correct spelling mistakes - input must be tokenized
test3 = Normalizer().correct_spelling_mistakes(test2)
print(test3)


# In[ ]:





# In[ ]:




