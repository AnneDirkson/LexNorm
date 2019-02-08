
# coding: utf-8

# Lexical normalization pipeline 
# 
# author - AR Dirkson 
# date - 2-10-2018
# 
# This pipeline takes raw text data and performs: 
# - Removes URLs, email addresses and personal pronouns
# - Convert to lower-case
# - Tokenization with NLTK
# - Removes non_English posts (conservatively) using langid module with top 10 languages and threshold of 100
# - British English to American English 
# - Normalization of generic abbreviations and slang 
# - Normalization of domain-specific (patient forum) abbreviations 
# - Spelling correction 
# 
# The spelling correction only works on a Linux platform.
# 

# All the relevant objects should be in obj_lex

# if you have not already done so you will need to import the names corpus
# 
# import nltk
# nltk.download('names')

# In[122]:


import pandas as pd 
import pickle 
import re
import numpy as np 
from collections import Counter, defaultdict, OrderedDict
from nltk import word_tokenize, pos_tag
import editdistance
import csv 
from sklearn.metrics import f1_score
import numpy as np
import scipy.stats 
from nltk.corpus import names

from weighted_levenshtein import lev, osa, dam_lev

import langid
from nltk.tokenize.treebank import TreebankWordDetokenizer


# In[127]:


class Normalizer (): 
        
    def __init__(self): 
        pass
        
    #to use this function the files need to be sorted in the same folder as the script under /obj_lex/
    def load_obj(self, name):
        with open('obj_lex/' + name + '.pkl', 'rb') as f:
            return pickle.load(f, encoding='latin1')
        
    def load_files(self): 
        self.ext_vocab2 = self.load_obj('vocabulary_spelling_unique')
        self.abbr_dict = self.load_obj ('abbreviations_dict')
        self.celex_freq_dict = self.load_obj ('celex_lwrd_frequencies')
        self.celex_list = list(self.celex_freq_dict.keys())
        self.celex_set = set (self.celex_list)
        self.drug_norm_dict = self.load_obj ('drug_normalize_dict')

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
            nw_msg = re.sub (r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", '-EMAIL-', msg) 
            list_of_msgs2.append(nw_msg)
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
    

    def create_names_list (self): 
        male_names = names.words('male.txt')
        female_names = names.words('female.txt')
        male_set = set (male_names)
        female_set = set (female_names)
        names_set = male_set.union(female_set) 
        names_list = [] 
        for word in names_set: 
            if (word != 'ned') & (word != 'Ned'): #ned means no evidence and is an important medical term
                word1 = str.lower (word)
                names_list.append(word1) #add the lowered words
                names_list.append(word) #add the capitalized words
        
        self.names_list = names_list
    
    def remove_propernoun_names(self,msg):
        try: 
            nw_msg = [self.change_tup_to_list(token) for token in msg]
            for a, token in enumerate (nw_msg):
                if (token[0] in self.names_list) and ((token[1] == 'NNP') or (token[1]== 'NNPS')): 
                    new_token = token[0].replace (token[0], "-NAME-")
                    nw_msg[a] = [new_token, token[1]]
#             nw_msg2 = [self.change_list_to_tup(token) for token in nw_msg]
            return nw_msg
        except TypeError: 
            pass
    
    
    def anonymize (self, posts): 
        posts2 = self.replace_urls (posts)
        posts3 = self.replace_email (posts2)
        posts4 = self.remove_empty(posts3)
        posts5 = [word_tokenize (sent) for sent in posts4]
        posts6 = [pos_tag(sent) for sent in posts5]
        self.create_names_list()
        posts7 = [self.remove_propernoun_names (m) for m in posts6]
        posts8 = []
        for post in posts7: 
            tg = [m[0] for m in post]
            posts8.append(tg)
        return posts8

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
        d = TreebankWordDetokenizer()
        posts2 = [d.detokenize(m) for m in posts]
        
        posts_temp, tally, list_removed = self.language_identify_basic(posts2)        
        lang = []

        for itm in list_removed: 
            lang.append(itm[1])

        c = Counter(lang)

        lang_list = ['en']

        for itm in c.most_common(10): 
            z = list(itm)
            lang_list.append(z[0])
    
        print("Most common 10 languages in the data are:" + str(lang_list))
        posts3, tally_nw, list_removed_nw = self.language_identify_thres(posts2, lang_list, thres = -100)
        return posts3
    
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
                    text_string += t_lower + ' '
            else:  # i.e., if the token is in the ignorelist..
                text_string += t_lower + ' '
            normalized_tokens = text_string.split()
        # print normalized_tokens
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
            if t in abbr.keys(): 
                nt = abbr[t]
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
    
    def load_files2 (self): 
        #load the edit matrices
        #transpositions
        self.edits_trans = self.load_obj ('weighted_edits_transpositions')
        #deletions 
        self.edits_del = self.load_obj('weighted_edits_deletions')
        #insertions 
        self.edits_ins = self.load_obj('weighted_edits_insertions')
        #substitutions
        self.edits_sub = self.load_obj('weighted_edits_substitutions')
                
        #load the generic dictionary - CHANGE PATH!  
        self.celex_freq_dict = self.load_obj ('celex_lwrd_frequencies')
    
    
    def initialize_weighted_matrices(self): 
    #initialize the cost matrixes for deletions and insertions
        insert_costs = np.ones(128, dtype=np.float64)  # make an array of all 1's of size 128, the number of ASCII characters
        delete_costs = np.ones (128, dtype=np.float64)

        for index,row in self.edits_ins.iterrows(): 
            insert_costs[ord(index)] = row['transformed_frequency']

        for index,row in self.edits_del.iterrows(): 
            delete_costs[ord(index)] = row['transformed_frequency']

        #substitution

        substitute_costs = np.ones((128, 128), dtype=np.float64)
        lst = []
        for index,row in self.edits_sub.iterrows(): 
            z = tuple([row['edit_from'], row['edit_to'], row['transformed_frequency']])
            lst.append (z)
        for itm in lst: 
            itm2 = list(itm)
            try: 
                substitute_costs[ord(itm2[0]), ord(itm2[1])] = itm2[2]
            except IndexError: 
                pass

        #transposition

        transpose_costs = np.ones((128, 128), dtype=np.float64)

        lst = []

        for index,row in self.edits_trans.iterrows(): 
            z = tuple([row['first_letter'], row['second_letter'], row['transformed_frequency']])
            lst.append (z)

        for itm in lst: 
            itm2 = list(itm)
            try: 
                transpose_costs[ord(itm2[0]), ord(itm2[1])] = itm2[2]
            except IndexError: 
                print(itm2)

        return insert_costs, delete_costs, substitute_costs, transpose_costs

    
    def weighted_ed_rel (self, cand, token, del_costs, ins_costs, sub_costs, trans_costs): 
        w_editdist = dam_lev(token, cand, delete_costs = del_costs, insert_costs = ins_costs, 
                             substitute_costs = sub_costs, transpose_costs = trans_costs)
        rel_w_editdist = w_editdist/len(token)
        return rel_w_editdist

    def run_low (self, word, voc, func, del_costs, ins_costs, sub_costs, trans_costs): 
        replacement = [' ',100]
        for token in voc: 
            sim = func(word, token, del_costs, ins_costs, sub_costs, trans_costs)
            if sim < replacement[1]:
                replacement[1] = sim
                replacement[0] = token

        return replacement   
    
    
    def spelling_correction (self, post, token_freq_dict, token_freq_ordered, min_rel_freq = 2, max_rel_edit_dist = 0.08): 
        post2 = []
        cnt = 0 

        for a, token in enumerate (post): 
            if self.TRUE_WORD.fullmatch(token):
                if token in self.spelling_corrections:
                    correct = self.spelling_corrections[token] 
                    post2.append(correct)
                    cnt +=1
                    self.replaced.append(token)
                    self.replaced_with.append(correct)

                elif token in self.celex_freq_dict:
                    post2.append(token)

                else:

                    # make the subset of possible candidates
                    freq_word = token_freq_dict[token]
                    limit = freq_word * min_rel_freq
                    subset = [t[0] for t in token_freq_ordered if t[1]>= limit]

                    #compare these candidates with the word        
                    candidate = self.run_low (token, subset, self.weighted_ed_rel, self.delete_costs_nw, self.insert_costs_nw, 
                                         self.substitute_costs_nw, self.transpose_costs_nw)

                #if low enough RE - candidate is deemed good
                    if candidate[1] <= max_rel_edit_dist:
                        post2.append(candidate[0]) 
                        cnt +=1
                        self.replaced.append(token)
                        self.replaced_with.append(candidate[0])
                        self.spelling_corrections [token] = candidate[0]
                    else: 
                        post2.append(token)
            else: post2.append(token)
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

    def create_token_freq (self, data): 
        flat_data = [item for sublist in data for item in sublist]
        self.token_freq = Counter(flat_data)
        
        token_freq_ordered = self.token_freq.most_common ()
        self.token_freq_ordered2 = [self.change_tup_to_list(m) for m in token_freq_ordered]
    
    def correct_spelling_mistakes(self, data): 
#         data= self.load_obj ('/data/dirksonar/Project1_lexnorm/spelling_correction/output/', 'gistdata_lemmatised')
        self.load_files2()
        self.insert_costs_nw, self.delete_costs_nw, self.substitute_costs_nw, self.transpose_costs_nw = self.initialize_weighted_matrices()
        self.total_cnt, self.replaced, self.replaced_with, self.spelling_corrections = self.initialize_files_for_spelling()
    
        self.TRUE_WORD = re.compile('[-a-z]+')  # Only letters and dashes  
#         data2 = [word_tokenize(m) for m in data]
        self.create_token_freq(data)
        out = [self.spelling_correction (m, self.token_freq, self.token_freq_ordered2) for m in data]
        return out, self.total_cnt, self.replaced, self.replaced_with, self.spelling_corrections
    
#--------Overall normalization function--------------------------------------

    def normalize (self, posts):
        self.load_files ()
        posts1 = self.anonymize(posts)
        posts2 = [self.lowercase (m) for m in posts1]
        posts3 = self.remove_non_english (posts2)
        posts3b = [word_tokenize(m) for m in posts3]
        posts4 = [self.sarker_normalize(posts3b)]
        posts5 = [self.expand_abbr(posts4[0], self.abbr_dict)]
        posts6, total_cnt, replaced, replaced_with, spelling_corrections = self.correct_spelling_mistakes(posts5[0])
        return posts6


# In[128]:


phony_posts = ['advil is good for relief. I like the colour', 'la vie est belle vous temps fait avec moi', 'Daniela works very hard and now has to have an op', 'lol testing is fun']


# In[130]:


p2 =Normalizer().normalize(phony_posts)
print(p2)


# In[90]:


print(p2)

