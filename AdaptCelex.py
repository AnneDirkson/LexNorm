
# coding: utf-8

# # Loading CELEX files and constructing working files
# 
# --author -- AR Dirkson 
# --created -- june 2018
# 
# Importing the CELEX files and using the sum of the frequencies (of different meaning for the same term e.g. fish (noun) and fish(verb)) to make a frequency dictionary. also all words are converted to lowercase and combined (e.g. AA and aa become aa with summed frequency)

# In[ ]:


import pickle


# In[1]:


# emw_file = "C:\\Users\\dirksonar\\Documents\\Stored_data\\data_from_others\\celex\\EMW.CD"
celex_file = FILL IN PATH

celex_list = []

with open(emw_file,'r') as celex_words:
    for line in celex_words:
        fields = line.rstrip().split('\\')
        word_id = fields[0]
        word = fields[1]
        freq = fields[2]
        lemma_id = fields[3]
        celex_list.append(fields)


# In[2]:


print(celex_list[0:10])


# In[3]:


print(type(fields))
print(fields)


# Taking the sum of the frequencies of the different versions of a word (eg. fish as noun and fish as verb) as the frequency

# In[5]:


#also need to remove repeats 
wordlist = set ([line[1] for line in celex_list])
print(len(wordlist))

# bunny = [line[1] for line in celex_list]
# print(bunny[:20])
# print(len(bunny))

from collections import defaultdict

frequencies = defaultdict(list)
lwrd_frequencies = defaultdict(list)

for line in celex_list: 
    word = line[1]
    word2 = word.lower ()
    freq = line[2]
    frequencies[word].append(int(freq))
    lwrd_frequencies[word2].append(int(freq))

for key in frequencies: 
    value2 = sum (frequencies[key])
    frequencies [key] = value2
    
for key in lwrd_frequencies: 
    value2 = sum (lwrd_frequencies[key])
    lwrd_frequencies [key] = value2


# In[6]:


print(frequencies ['a'])

celex_frequencies = frequencies

print(lwrd_frequencies['a'])


# In[7]:


print(len(frequencies))
print(len(lwrd_frequencies)) #this makes sense

print ('Difference is: ' + str(len(frequencies) - len(lwrd_frequencies)))


# In[8]:


celex_frequencies2 = defaultdict (int)
for key in celex_frequencies: 
    celex_frequencies2 [key] = celex_frequencies[key]


# In[9]:


print(celex_frequencies2)


# In[10]:


print(celex_frequencies2 ['a'])


# In[11]:


celex_lwrd_frequencies = defaultdict (int)
for key in lwrd_frequencies: 
    celex_lwrd_frequencies [key] = lwrd_frequencies[key]


# In[12]:


print(celex_lwrd_frequencies ['a'])


# In[21]:


celex_lwrd_unique = list (celex_lwrd_frequencies.keys())

print(celex_lwrd_unique[0:10])


# Save both the CELEX wordlist and the CELEX freq dict

# In[13]:


def save_obj(obj, name):
        with open('obj/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(celex_lwrd_frequencies, 'celex_lwrd_frequencies')

save_obj(celex_lwrd_unique, 'celex_lwrd_unique')           

