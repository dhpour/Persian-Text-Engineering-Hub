# Persian text engineering legacy tools and libraries
[![Check Links](https://github.com/dhpour/Persian-Text-Engineering-Hub/actions/workflows/link_check.yml/badge.svg)](https://github.com/dhpour/Persian-Text-Engineering-Hub/actions/workflows/link_check.yml)

Curated list of Text libraries, tools and datasets for Persian language.

In each section, tools, libraries, models, and datasets related to the main topic of that section are listed.

For datasets check >> [Datasets](DATASETS.md)

For any other resources check >> [Other](OTHER.md)

Topics
------
- [Multi-purpose libs](#multi-purpose-libs)
- [Grapheme to Phoneme](#graheme-to-phoneme)
- [Word Analyzing](#word-analyzing)
- [Sentiment Analysis](#sentiment-analysis)
- [Informal Persian](#informal-persian)
- [Numbers <> Words](#numbers--words)
- [Dependency Parsing](#dependency-parsing)
- [Spell Checking](#spell-checking)
- [Normalization](#normalization)
- [Transliteration](#transliteration)
- [Crawl Suite](#crawl-suite)
- [POS Tagging](#pos-tagging)
- [Various](#various)
- [Mocking](#mocking)
- [UI/UX](#uiux)
- [Generation](#generation)

Multi-purpose libs
------------------
### [Parsivar](https://github.com/ICTRC/Parsivar)
A Language Processing Toolkit for Persian
- Normalizer / Tokenizer (sentences / words)
- Stemmer
- POS Tagger
- Chunker
- Dependency Parser
- Spell Checker

### [Hazm](https://github.com/roshan-research/hazm)
Persian NLP Toolkit
- Normalizer / Tokenizer
- Lemmatizer
- POS Tagger
- Chunker
- Dependency Parser
- Word / Sentence Embedding
- Different Corpora reader

### [Hezar](https://github.com/hezarai/hezar)
The all-in-one AI library for Persian, supporting a wide variety of tasks and modalities!
- POS Tagger
- Text Classification (sentiment analysis, categorization, etc)
- Sequence Labeling (POS, NER, etc.)
- Mask Filling
- Speech Recognition
- Text Detection
- Image to Text (OCR)
- Image to Text (License Plate Recognition)
- Image to Text (Image Captioning)
- Word Embeddings
  - FastText
  - Word2Vec (Skip-gram)
  - Word2Vec (CBOW)
- Datasets

### [polyglot](https://github.com/aboSamoor/polyglot)
Multilingual text (NLP) processing toolkit. Consists of some useful Persian functionalities:
- Tokenizer (Sentence / Word)
- Named Entity Recognition
- Morpheme Extractor
- Language Detector


Graheme to phoneme
------------------
### [Persian Phonemizer](https://github.com/de-mh/persian_phonemizer)

A tool for translating Persian text to IPA (International Phonetic Alphabet).

### [G2P Fa](https://github.com/de-mh/g2p_fa)
A Grapheme to Phoneme model using LSTM implemented in pytorch

### [PersianG2P](https://github.com/PasaOpasen/PersianG2P)
Persian Grapheme-to-Phoneme (G2P) converter

### [Persian Words Pronunciation](https://github.com/pfndesign/persian-words-pronunciation)
list of persian word pronunciations

### [Persian text to speech](https://github.com/AlisterTA/Persian-text-to-speech)
It is a convolutional sequence to sequence model created based on [Tachibana et al](https://arxiv.org/abs/1710.08969) with modifications. This repo consists of notebooks to do the training and inferencing and provides proper datasets to do so.

### [Persian_g2p: A seq-to-seq model for Persian G2P mapping](https://github.com/AzamRabiee/Persian_G2P)
Persian Grapheme-to-Phoneme (G2P) converter

### [G2P](https://github.com/mohamad-hasan-sohan-ajini/G2P)
The G2P algorithm is used to generate the most probable pronunciation for a word not contained in the lexicon dictionary. It could be used as a preprocess of text-to-speech system to generate pronunciation for OOV words.

### [Tihu Persia Dictionary](https://github.com/tihu-nlp/tihudict)
Tihu-dict is a pronouncing dictionary of Persian

Word Analyzing
-------------
### [CPIA - Contemporary Persian Inflectional Analyzer](https://github.com/dhpour/cpia)
Informal and Formal Persian word analyzer (inflection with FST)

### [Persian Morphologically Segmented Lexicon 0.5](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3011)
This dataset includes 45300 Persian word forms which are manually segmented into sequences of morphemes.

### [Universal Derivations v1.1](https://github.com/lukyjanek/universal-derivations/tree/master/fa/DeriNetFA)
Universal Derivations (UDer) is a collection of harmonized lexical networks capturing word-formation, especially derivation, in a cross-linguistically consistent annotation scheme for many languages including `Persian` (semi-automatically). Consists of 7k families, 43k lexemes and 35k relations. [Article](https://aclanthology.org/W19-8511.pdf). [Dataset files](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3247/UDer-1.1.tgz).

### [polyglot](https://github.com/aboSamoor/polyglot)
A morpheme Extracter for 135 languages including `Persian`.

### [PARSEME Corpse Fa](https://gitlab.com/parseme/parseme_corpus_fa)
PARSEME is a verbal multiword expressions (VMWEs) corpus for Farsi. All the annotated data come from a subset of the Farsi section of the [MULTEXT-East "1984"](https://nl.ijs.si/ME/Vault/V4/) annotated corpus 4.0. More than colums of LEMMA UPOS, XPOS, FEATS, HEAD and DEPREL there is also PARSEME:MVE which is manually annotated.

### [Universal Segmentations](https://ufal.mff.cuni.cz/universal-segmentations)
Universal Segmentations (UniSegments) is a collection of lexical resources capturing morphological segmentations harmonised into a cross-linguistically consistent annotation scheme for many languages including `Persian`. The annotation scheme consists of simple tab-separated columns that stores a word and its morphological segmentations, including pieces of information about the word and the segmented units, e.g., part-of-speech categories, type of morphs/morphemes etc. It also has a [python library](https://github.com/ufal/universal-segmentations) or creating such data from text. This dataset consists of 45k Persian words.

### [Perstem](https://github.com/jonsafari/perstem)
Persian stemmer and morphological analyzer

### [Persian Stemming Dataset](https://github.com/htaghizadeh/PersianStemmingDataset/tree/master)
Consists of two stemmeing sets. 1) 4k words from [Bootstrapping the Development of an HPSG-based Treebank for Persian](https://journals.colorado.edu/index.php/lilt/article/view/1301/1133) and 2) 27k words from [A syntactic valency lexicon for Persian verbs : The first steps towards Persian dependency treebank](https://www.researchgate.net/profile/Mohammad-Sadegh-Rasooli/publication/230612993_A_Syntactic_Valency_Lexicon_for_Persian_Verbs_The_First_Steps_towards_Persian_Dependency_Treebank/links/0912f50251db69938a000000/A-Syntactic-Valency-Lexicon-for-Persian-Verbs-The-First-Steps-towards-Persian-Dependency-Treebank.pdf).

### [Persian Stemmer Python](https://github.com/htaghizadeh/PersianStemmer-Python)
A stemmer for Persian based on [A new hybrid stemming method for persian language](https://github.com/htaghizadeh/PersianStemmer-Python)

Sentiment Analysis
------------------
### [Persian Sentiment Resources](https://github.com/Text-Mining/Persian-Sentiment-Resources)
Awesome Persian Sentiment Analysis Resources - منابع مرتبط با تحلیل احساسات در زبان فارسی

- Consists of following datasets:
  - Deep Neural Networks in Persian Sentiment Analysis
  - Sentiment Analysis Challenges
  - Sentiment Lexicon
  - Sentiment Tagged Corpus (dataset)
  - HesNegar: Persian Sentiment WordNet

### [Persian Sentiment Analyzer](https://github.com/ashalogic/Persian-Sentiment-Analyzer)
Consists of data (3K) and code (notebook) to create a LSTM model for Sentiment Analysis.

### [Sentiment Analysis](https://github.com/parsa-abbasi/Sentiment-Analysis)
Sentiment analysis using ML and DL models on Persian texts

### [LexiPers](https://github.com/phosseini/LexiPers)
A Sentiment Analysis Lexicon for Persian. Consists of 4k words

### [Taaghche | طاقچه](https://www.kaggle.com/datasets/saeedtqp/taaghche)
Persian book comment ratings dataset. Consists of about 70k comment about 11k books.

### [Digikala (comments & products)](https://www.kaggle.com/datasets/radeai/digikala-comments-and-products)
The Digikala (comments & products) dataset offers a comprehensive glimpse into the vast online marketplace of Digikala, comprising over 1.2 million products and more than 6 million comments.

### [Digikala Comments](https://www.kaggle.com/datasets/soheiltehranipour/digikala-comments-persian-sentiment-analysis)
3k comments with score and ratings.

### [MirasOpinion](https://github.com/miras-tech/MirasText/tree/master/MirasOpinion)
93k digikala products comments with manual labeling.

### [Persian tweets emotional dataset](https://www.kaggle.com/datasets/behdadkarimi/persian-tweets-emotional-dataset)
20k tweets with emotion identification labels.

### [Persian Emotion Detection (tweets)](https://github.com/nazaninsbr/Persian-Emotion-Detection)
A Dataset of 30,000 emotion labeled Persian Tweets.

### [Persian Text Emotion](https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion)
Consists of 5.56K tweets with labels (sadness, anger, happiness, hatred, wonder and fear) describing their emotions.

### [ArmanEmo](https://github.com/Arman-Rayan-Sharif/arman-text-emotion)
Consists of 7k docs with 6 emotion label types (sadness, anger, happiness, hatred, wonder, fear).

### [Snappfood](https://hooshvare.github.io/docs/datasets/sa#snappfood)
Snappfood (an online food delivery company) user comments containing 70,000 comments with two labels (i.e. polarity classification): Happy, Sad.

### [NRC Persian Lexicon](https://github.com/mhbashari/NRC-Persian-Lexicon)
It is the Persian translation of NRC Emotion Lexicon which is a list of English words with their associate basic emotions in eigth categories( anger, fear, anticipation, trust, surprise, sadness, joy, and disgust).

### [Pars ABSA](https://github.com/Titowak/Pars-ABSA)
Consists of 10k samples which each record focuses on one aspect (e.g. camera, screen resolution, etc of a comment about a cell phone) of a comment. Each comment may appear on more than one sample based on the number of aspects that exist in that comment.

### [PerSent -- Persian Sentiment Analysis and Opinion Mining Lexicon](https://www.gelbukh.com/resources/persent/)
Consists of 1500 words with their degrees of polarity.

### [DeepSentiPers](https://github.com/JoyeBright/DeepSentiPers)
Utilizes the [SentiPers dataset](https://github.com/phosseini/sentipers), which consists of 7,400 sentences, and enhances it with various embeddings to develop both LSTM and CNN models. All the original and newly transformed data, along with the notebooks used to create the models, are available in this repository.

### [ParsBERT](https://github.com/hooshvare/parsbert)
Fine-tuned a BERT based transofrmer on various sentiment analysis datasets like Digikala, SnappFood, SentiPers and Taaghche.

### [ParsiNLU](https://github.com/persiannlp/parsinlu)
Persian NLP team trained various mt5 models on their sentiment analysis dataset.

Informal Persian
----------------

### [Shekasteh](https://github.com/rasoolims/Shekasteh)
Shekasteh is an evaluation dataset for Persian colloquial text. It comes from different genres, including blog posts, movie subtitles, and forum chats.

### [CPIA](https://github.com/dhpour/cpia)
Informal and Formal Persian word analyzer (inflection with FST)

### [Persian Slang](https://github.com/semnan-university-ai/persian-slang)
Persian Slang Words (dataset)

### [Informal Persian Universal Dependency Treebank (iPerUDT)](https://github.com/royakabiri/iPerUDT)
Informal Persian Universal Dependency Treebank, consisting of 3000 sentences and 54,904 tokens, is an open source collection of colloquial informal texts from Persian blogs.

Numbers <> Words
----------------

### [NumToPersian](https://github.com/Shahnazi2002/NumToPersian)
Converts numbers to words.

### [Convert numbers to Persian words](https://github.com/saeed-raeisi/num2words)
Read me this number python -- Convert number to Persian

### [PersianNumberToWord](https://github.com/razavioo/PersianNumberToWord)
Convert numbers to Persian words.

### [DPERN](https://github.com/amishbni/dpern)
Describe PERsian Numbers

### [ParsiNorm](https://github.com/haraai/ParsiNorm)
A normalizer which do a lot about numbers, both ways.

### [Persian Tools](https://github.com/persian-tools/py-persian-tools)
Handling various number types in Persian text (like National ID, Sheba, etc)

### [petit](https://github.com/JKhakpour/petit)
Persian text -> integer, ineteger -> text converter

### [num2fawords](https://github.com/5j9/num2fawords)
Takes a number and converts it to Persian word form

Dependency Parsing
------------------
### [The Persian Universal Dependency Treebank (Persian UD)](https://github.com/UniversalDependencies/UD_Persian-Seraji)
The Persian Universal Dependency Treebank (Seraji) is based on Uppsala Persian Dependency Treebank (UPDT). The conversion of the UPDT to the Universal Dependencies was performed semi-automatically with extensive manual checks and corrections.

### [The Persian Universal Dependency Treebank (PerUDT) (v1.0)](https://github.com/UniversalDependencies/UD_Persian-PerDT)
The Persian Universal Dependency Treebank (PerUDT) is the result of automatic coversion of Persian Dependency Treebank (PerDT) with extensive manual corrections. Consists of 29k sentences.

### [PARSEME Corpse Fa](https://gitlab.com/parseme/parseme_corpus_fa)
PARSEME is a verbal multiword expressions (VMWEs) corpus for Farsi. All the annotated data come from a subset of the Farsi section of the [MULTEXT-East "1984"](https://nl.ijs.si/ME/Vault/V4/) annotated corpus 4.0. More than colums of LEMMA UPOS, XPOS, FEATS, HEAD and DEPREL there is also PARSEME:MVE which is manually annotated.

### [UDPipe 2](https://github.com/ufal/udpipe/tree/udpipe-2)
UDPipe: Trainable pipeline for tokenizing, tagging, lemmatizing and parsing Universal Treebanks and other CoNLL-U files.

### [Informal Persian Universal Dependency Treebank (iPerUDT)](https://github.com/royakabiri/iPerUDT)
Informal Persian Universal Dependency Treebank, consisting of 3000 sentences and 54,904 tokens, is an open source collection of colloquial informal texts from Persian blogs.

Spell checking
------------

### [Persian Spell Checker with Kenlm](https://github.com/pooya-mohammadi/persian-spell-checker-kenlm)
A complete instruction for training a Persian spell checker and a language model based on SymSpell and KenLM, using Wikipedia dataset. Tokens that are not in the vocab and has a very low frequency considered to be miss-spelled words and replaced with their equivalent from vocabs which maximizes the probabilty of the sentence.

### [FAspell](https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-1547)
FASpell dataset was developed for the evaluation of spell checking algorithms. It contains a set of pairs of misspelled Persian words and their corresponding corrected forms similar to the ASpell dataset used for English.
The dataset consists of two parts: a) faspell_main: list of 5050 pairs collected from errors made by elementary school pupils and professional typists. b) faspell_ocr: list of 800 pairs collected from the output of a Farsi OCR system.

### [Lilak, Persian Spell Checking Dictionary](https://github.com/b00f/lilak)
Created data for [hunspell library](https://github.com/hunspell/hunspell) for spell checking and morphology analyzing.

### [Persian Spell Checker](https://github.com/reza1615/Persian-Spell-checker)
Consists of some lists of miss-spelled words and some dictionaries of Persian word entries.

### [PerSpellData](https://github.com/rominaoji/PerSpellData)
A comprehensive parallel dataset designed for the task of spell checking in Persian. Misspelled sentences together with the correct form are produced using a massive confusion matrix, which is gathered from many sources. This dataset contains informal sentences in addition to the formal sentences, and contains texts from diverse topics. Both non-word and real-word errors are collected in the dataset

### [HeKasre](https://github.com/AUT-Data-Group/HeKasre)
Code and data for detecting and correcting just a special kind of cognitive miss-spelling error in `informal Persian`.

Normalization
-------------
### [PersianUtils](https://github.com/ishto7/persianutils)
Standardize your Persian text: Preprocessing, Embedding, and more!

### [Farsi-Normalizer](https://github.com/dhpour/Farsi-Normalizer)
Simple Farsi normalizer

### [virastar](https://github.com/aziz/virastar)
Cleanning up Persian text! (Ruby)
- [Python version](https://github.com/JKhakpour/virastar.py)

### [Virastar (ویراستار)](https://github.com/brothersincode/virastar)
Virastar is a Persian text cleaner (JS).
- [PHP version 1](https://github.com/octoberfa/virastar)
- [PHP version 2](https://github.com/AlirezaSedghi/Virastar)
- [TypeScript version](https://github.com/nekofar/virastar)

### [Farsi Analyzer](https://github.com/NarimanN2/ParsiAnalyzer)
A Persian normalization and tokenization tool, constructed as a plugin for Elasticsearch.

### [ParsiNorm](https://github.com/haraai/ParsiNorm)
A normalizer which do a lot about numbers, both ways.

Transliteration
---------------
### [Tajik-to-Persian transliteration](https://github.com/stibiumghost/tajik-to-persian-transliteration)
Tajik-to-Persian transliteration model

### [F2F](https://github.com/brothersincode/f2f)
Farsi to Finglish, a Persian transliterator

### [Behnevis](https://github.com/dhpour/pinglish_behnevis)
24k ASCII transliterated Persian words

### [Farsi to Tajiki](https://github.com/kor-gar/farsi)
An attempt to make a transliterator of Farsi (Persian) web page to Tajiki (Cyrillic) with a bookmarklet.

Crawl Suite
-----------
### [Persian News Search Engine](https://github.com/MehranTaghian/news-search-engine/tree/main)
A search engine for crawling news from the web, storing in a structured way, and querying through the stored documents for finding the most relevant results using Machine Learning and Information Retrieval techniques.

### [iranian-news-agencies-crawler](https://github.com/hamid/iranian-news-agencies-crawler)
a crawler to fetch last news from Iranian(Persian) news agencies.

### [PersianCrawler](https://github.com/pourmand1376/PersianCrawler)
Open source crawler for Persian websites including Asriran, fa-Wikipedia, Tasnim, Isna.

POS Tagging
----------
### [Persian_POS_Tagger](https://github.com/AminMozhgani/Persian_POS_Tagger)
A Persian POS Tagger trained by The Persian Universal Dependency Treebank (Persian UD) with Tensorflow

### [PARSEME Corpse Fa](https://gitlab.com/parseme/parseme_corpus_fa)
PARSEME is a verbal multiword expressions (VMWEs) corpus for Farsi. All the annotated data come from a subset of the Farsi section of the [MULTEXT-East "1984"](https://nl.ijs.si/ME/Vault/V4/) annotated corpus 4.0. More than colums of LEMMA UPOS, XPOS, FEATS, HEAD and DEPREL there is also PARSEME:MVE which is manually annotated.

### Multi-purpose tools with POS Tagging capability
- [Parsivar](https://github.com/ICTRC/Parsivar)

- [Hazm](https://github.com/roshan-research/hazm)

- [Hezar](https://github.com/hezarai/hezar)

### [Farsi NLP Tools](https://github.com/wfeely/farsiNLPTools)
Scripts and models developed for POS Tagging and Dependency Parsing Persian based on [TurboParser](http://www.ark.cs.cmu.edu/TurboParser).

### [RDR POS Tagger](https://github.com/datquocnguyen/RDRPOSTagger)
RDRPOSTagger is supports pre-trained UPOS, XPOS and morphological tagging models for about 80 languages including `Persian`. [Java version](https://github.com/datquocnguyen/jPTDP).

### [Cross-platform Persian Parts-of-Speech tagger](https://github.com/mhbashari/perpos)
This is another persian POS tagger

Various
------------------
### [Perke](https://github.com/AlirezaTheH/perke)
A keyphrase extractor for Persian

### [PREDICT-Persian-Reverse-Dictionary](https://github.com/arm-on/PREDICT-Persian-Reverse-Dictionary)
The first intelligent Persian reverse dictionary. Consists of various models for this task and [datasets](https://www.kaggle.com/datasets/malekzadeharman/persian-reverse-dictionary-dataset) of Amid, Moeen, Dehkhoda, Persian Wikipedia and Persian Wordnet ([Farsnet](http://farsnet.nlp.sbu.ac.ir/)).

### [Persian-ATIS (Airline Travel Information System) Dataset](https://github.com/Makbari1997/Persian-Atis)
A Persian dataset for Joint Intent Detection and Slot Filling.

### [ParsiNLU Reading Comprehension](https://github.com/persiannlp/parsinlu)
Persian NLP team trained various mt5 models on their reading comprehension dataset.

Mocking
-------
### [PersianFaker](https://github.com/muhammadmp97/PersianFaker)
Do you need some fake data?

UI/UX
-----
### [Persian-Badge](https://github.com/kasraghoreyshi/persian-badge)
Persian-Badge is a website for having metadata badges in the Persian language

Generation
----------

### [Dorna Llama3 8B Instruct](https://huggingface.co/PartAI/Dorna-Llama3-8B-Instruct)
The Dorna models are a family of decoder-only models, specifically trained/fine-tuned on Persian data. This model is built using the Meta Llama 3 Instruct model. There are also [quantized versions](https://huggingface.co/PartAI/Dorna-Llama3-8B-Instruct-GGUF) of this model.

### [PersianLLaMA 13B Instruct](https://huggingface.co/ViraIntelligentDataMining/PersianLLaMA-13B-Instruct)
With 13 billion parameters, this model has been fine-tuned using the [Persian Alpaca dataset](https://huggingface.co/datasets/sinarashidi/alpaca-persian) on Lllama 2 to excel at executing detailed instructions and delivering tailored outputs. There is also [PersianLLaMA 13B](https://huggingface.co/ViraIntelligentDataMining/PersianLLaMA-13B) which is fine-tuned on Persian wikipedia.

### [ParsGPT](https://github.com/hooshvare/parsgpt)
Persian version of GPT2 model fine-tuned on Persian poetry and [ParsiNLU sentiment analysis](https://github.com/persiannlp/parsinlu) datast.

### [AVA LLM collection](https://huggingface.co/collections/MehdiHosseiniMoghadam/ava-6648848e7a1ed3e0016f8395)
Fine-tuned versions of Mistral 7B and Llama 3 for Persian. The Persian resources used for these models are not known.

### [MaralGPT](https://huggingface.co/MaralGPT)
Is a Mistral 7B based model trained on [Alpaca Persian Instruction dataset](https://huggingface.co/datasets/sinarashidi/alpaca-persian).

Thanks
------
Thanks to [Awesome Persian NLP](https://github.com/mhbashari/awesome-persian-nlp-ir) and [Awesome Iranian Datasets](https://github.com/MEgooneh/awesome-Iran-datasets) for providing some elements of this long list.
