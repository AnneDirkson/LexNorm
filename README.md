# LexNorm

Lexical normalization pipeline for medical patient forum data (Python 3). 

author - AR Dirkson 

date - 15-07-2019

last update -- 30-9-2019

This pipeline takes raw text data and performs: 
- Removes URLs, email addresses and personal pronouns 
(optional, default = False, change anonymize to True in Normalizer().normalize (text))
- Tokenization with NLTK 
- Remove non-English posts (conservatively) using langid [3] 
(optional, default = False, change remove_foreign to True in Normalizer().normalize(text))
- British English to American English 
- Normalization of contractions
- Normalization of generic abbreviations and slang 
- Normalization of domain-specific (patient forum) abbreviations 

- Spelling correction * 

* In a seperate function: Normalizer().correct_spelling_mistakes() 

For more detail on the pipeline see: 

Dirkson AR, Verberne S, Sarker A and Kraaij W (2019). Data-driven Lexical Normalization for Medical Social Media. Multimodal Technologies and Interaction, 3(3): 60. 

See: https://www.mdpi.com/2414-4088/3/3/60

Please refer to our article if you use this module.

# Updates after article release

- Drug names based on the FDA database of drugs and their active ingredients are now excluded from spelling correction to prevent common drug names replacing uncommon, similar drug names. 

(Data downloaded from: https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files) 

- There is now the possibility to use a different token frequency than that of the data you are preprocessing for spelling correction. This may be useful if you have a larger dataset available than the one you are preprocessing. You need to use the different_token_freq = True option in the Normalizer().correct_spelling_mistakes() i.e. Normalizer().correct_spelling_mistakes(data, different_token_freq = True). Beforehand you will need to create the token frequency yourself and save it under obj_lex/token_freq.pkl

To make a token frequency, you will need to use the Counter() on all the words in the data (also see the create_token_freq() function for help).

- Added function for normalizing ’ to '. This was messing with the contraction expansion. 

# Required files: 
Prior to running this normalizer you will need to download the tetragram_model.binary in the N-gram-language-models file at https://data.mendeley.com/datasets/dwr4xn8kcv/3. These models have been developed by Abeed Sarker and Graciela Gonzalez- Hernandez [1]. This model is too large to add to the GitHub repository. You should save this model in the obj_lex folder.

The necessary in-house created abbreviations_dict is a dictionary of the domain-specific abbreviations, created based on a rare cancer forum together with a domain expert by our group. 

The following files are provided in the obj_lex folder and have been provided by other researchers:

- This script makes use of the HealthVec model developed by Miftahutdinov et al. [4] which has been downloaded from: https://github.com/dartrevan/ChemTextMining/tree/master/word2vec

- no_slang_mod.txt, english_spellings.txt, 1_2letter_words.txt and american_spellings.txt are from Sarker et al. [2]. Can also be found at https://bitbucket.org/asarker/simplenormalizerscripts

- The aspell_dict_lower is a lowered version of word list 60 of the publicly available GNU Aspell dictionary. See: http://aspell.net/

References: 

[1] A. Sarker & G. Gonzalez-Hernandez, 2017. A corpus for mining drug-related knowledge from Twitter chatter: Language models and their utilities. Data in Brief, 10.

[2] A. Sarker, 2017. A customizable pipeline for social media text normalisation. Social network analysis and mining, 7, 1.

[3] M. Liu & T. Baldwin, 2012. langid.py: An Off-the-shelf Language Identification Tool. Proceedings of the 50th annual meeting of the association of computational linguistics, p 25-30.

[4] Miftahutdinov, Z. S., Tutubalina, E. V., & Tropsha, A. E. (2017). Identifying disease-related expressions in reviews using conditional random fields. Proceedings of the International Conference Dialogue 2017. 

# Additional notes on spelling correction 

This script includes an unsupervised spelling correction module that uses the corpus to construct a list of plausible correction candidates based on relative corpus frequency (min 9x more frequent) and edit distance threshold (max of 0.76). If there are no plausible candidates, the token is not corrected.  The correction algorithm is a combination of relative Levenshtein distance (weight = 0.4) and language model similarity based on the HealthVec model (weight = 0.6). This language model is based on around 2.5M comments from online patient fora and can be substituted by a domain-specific distributed language model relevant to your domain. 

A decision process is used to determine if a word is a spelling mistake. The misspelling detection makes use of the ASPELL dictionary. This can be substituted by another generic dictionary. It is only used to determine if a word should not be corrected because it is a generic word.

The grid used for tuning the corpus frequency multiplier threshold was [0-10] (steps of 1). 
The grid used for the spelling mistake detection was [0.40-0.80] (steps of 0.02) for the maximum allowed relative edit distance. 

Note: if you only use the spelling correction function of the Normalizer script, the data must be tokenized.
