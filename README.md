# LexNorm

Lexical normalization pipeline for medical patient forum data. This repository contains the pipeline script for Jupyter Notebook (python 3). Updated 08-02-2019. 

This pipeline includes both the DrugNorm and SpellingCorrect modules. 

author - AR Dirkson 
date - 2-10-2018

This pipeline takes raw text data and performs: 
- Removes URLs, email addresses and personal pronouns
- Convert to lower-case
- Tokenization with NLTK 
- British English to American English 
- Normalization of generic abbreviations and slang 
- Normalization of domain-specific (patient forum) abbreviations 
- Spelling correction 

# Required files: 
The necessary in-house created abbreviations_dict is a dictionary of the domain-specific abbreviations, created based on a rare cancer forum together with a domain expert.

celex_lwrd_frequencies is an adapted version of the CELEX [1] with frequencies of different word senses combined and also lowered and capitalized variants combined. You can use another generic dictionary. Due to licensing, we cannot share but our adapted CELEX can be recreated with the AdaptCelex script.

no_slang_mod.txt, english_spellings.txt and american_spellings.txt are from Sarker et al. [2]. Can also be found at https://bitbucket.org/asarker/simplenormalizerscripts

The drug normalization dictionary 'drug_normalize_dict' is an adapted version of the RXNorm database (part of UMLS). All materials for creating this dictionary are under the DrugNorm repository. 

References: 
[1] G. Burnage, R.H. Baayen, R. Piepenbrock and H. van Rijn. 1990. CELEX: A Guide for Users.
[2] A. Sarker, 2017. A customizable pipeline for social media text normalisation. Social network analysis and mining, 7, 1.

# notes on spelling correction 

This script includes spelling correction module that uses unsupervised data to construct a list of candidates. The correction algorithm is a weighted Levenshtein distance algorithm. A decision process is used to determine if a word is a spelling mistake.

It makes use of the CELEX generic dictionary but this can be substituted by another generic dictionary. It is only used to determine if a word should not be corrected because it is a generic word.

The matrices for edit weights are included (but change the paths!) and due to licensing, CELEX is not.

The grid used for the spelling mistake detection was [0.05 - 0.15] (steps of 0.01) for relative weighted edit distance max and [2-10] (steps of 1) for relative corpus frequency multiplier. F0.5 was used as a metric. This can be re-tuned (tuning not included in this script).

Note: the damlev module only works on Linux platforms and the input data needs to be tokenized
