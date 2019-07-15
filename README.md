# LexNorm

Lexical normalization pipeline for medical patient forum data (Python 3). 

author - AR Dirkson 
date - 15-07-2019

This pipeline takes raw text data and performs: 
- Removes URLs, email addresses and personal pronouns
- Convert to lower-case
- Tokenization with NLTK 
- Remove non-English posts (conservatively) using langid [3]
- British English to American English 
- Normalization of contractions
- Normalization of generic abbreviations and slang 
- Normalization of domain-specific (patient forum) abbreviations 
- Spelling correction 

For more detail on the pipeline see: 

Dirkson AR, Verberne S, Sarker A and Kraaij W (Under submission). Data-driven Lexical Normalization for Medical Social Media. Multimodal Technologies and Interaction [Text Mining in Complex Domains].

Please refer to our article if you use this module.

# Required files: 
Prior to running this normalizer you will need to download the tetragram.binary and trigram.binary models in the N-gram-language-models file at https://data.mendeley.com/datasets/dwr4xn8kcv/3. These models have been developed by Abeed Sarker and Graciela Gonzalez- Hernandez [1]

The necessary in-house created abbreviations_dict is a dictionary of the domain-specific abbreviations, created based on a rare cancer forum together with a domain expert.

no_slang_mod.txt, english_spellings.txt, 1_2letter_words.txt and american_spellings.txt are from Sarker et al. [2]. Can also be found at https://bitbucket.org/asarker/simplenormalizerscripts

The aspell_dict_lower is a lowered version of word list 60 of the publicly available GNU Aspell dictionary. See: http://aspell.net/

References: 
[1] A. Sarker & G. Gonzalez-Hernandez, 2017. A corpus for mining drug-related knowledge from Twitter chatter: Language models and their utilities. Data in Brief, 10.
[2] A. Sarker, 2017. A customizable pipeline for social media text normalisation. Social network analysis and mining, 7, 1.
[3] M. Liu & T. Baldwin, 2012. langid.py: An Off-the-shelf Language Identification Tool. Proceedings of the 50th annual meeting of the association of computational linguistics, p 25-30.

# Additional notes on spelling correction 

This script includes an unsupervised spelling correction module that uses the corpus to construct a list of plausible correction candidates based on relative corpus frequency (min 9x more frequent) and edit distance threshold (max of 0.68). If there are no plausible candidates, the token is not corrected.  The correction algorithm is a combination of relative Levenshtein distance (weight = 0.4) and the probability of the trigram occuring according to a sequential trigram model (weight = 0.6). This language model was based on health-related Twitter data and can be substituted by a domain-specific trigram model relevant to your domain. 

A decision process is used to determine if a word is a spelling mistake. The misspelling detection makes use of the ASPELL dictionary. This can be substituted by another generic dictionary. It is only used to determine if a word should not be corrected because it is a generic word.

The grid used for tuning the corpus frequency multiplier threshold was [0-10] (steps of 1). 
The grid used for the spelling mistake detection was [0.40-0.80] (steps of 0.01) for the maximum allowed relative edit distance. 

Note: if you only use the spelling correction function of the Normalizer script, the data must be tokenized.
