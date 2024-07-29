# Persian-Text-Engineering-Hub
[![Check Links](https://github.com/dhpour/Persian-Text-Engineering-Hub/actions/workflows/link_check.yml/badge.svg)](https://github.com/dhpour/Persian-Text-Engineering-Hub/actions/workflows/link_check.yml)

Curated list of Text libraries, tools and datasets for Persian language.

Topics
------
- [Multi-purpose libs](#multi-purpose-libs)
- [Grapheme to Phoneme](#graheme-to-phoneme)
- [Word Analyzing](#word-analyzing)
- [Sentiment Analysis](#sentiment-analysis)
- [Informal Persian](#informal-persian)
- [Numbers <> Words](#numbers--words)
- [Word Embeddings](#embeddings)
- [Benchmark](#benchmark)
- [QA](#qa)
- [Dependency Parsing](#dependency-parsing)
- [Entailment](#entailment)
- [Datasets (classification)](#datasets-classification)
- [NER](#ner)
- [Unlabled and Raw Text](#unlabled-and-raw)
- [Toxic Text](#toxic-text)
- [Stop Word List](#stop-word-list)
- [Spell Checking](#spell-checking)
- [Normalization](#normalization)
- [Transliteration](#transliteration)
- [Encyclopedia and Word Set](#encyclopedia-and-word-set)
- [Poetry and Literature](#poetry-and-literature)
- [Audio Dataset](#audio)
- [Crawl Suite](#crawl-suite)
- [POS Tagging](#pos-tagging)
- [Various](#various)
- [Base Models](#base-models)
- [Mocking](#mocking)
- [UI/UX](#uiux)
- [OCR](#ocr)
- [Spam](#spam)
- [Image Captioning](#image-captioning)
- [Translation](#translation)
- [Knowledge Graph](#knowledge-graph)
- [Summery](#summary)
- [Paraphrase](#paraphrase)
- [WSD](#wsd)
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
Uses [SentiPers](https://github.com/phosseini/sentipers) data which contains 7400 sentences and extends it with different embeddings for creating both LSTM and CNN models. All old and new transformed data and notebooks to create models are available in this repo.

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

Embeddings
---------------

### [FastText](https://fasttext.cc/docs/en/crawl-vectors.html)
Pre-trained word vectors of 157 languages including `Persian`, trained on CommonCrawl and Wikipedia using CBOW.

### [Persian Word Embedding](https://github.com/miladfa7/Persian-Word-Embedding)
A tutorial on how to use 3 word embeddings; a) Downloading and using fasttext Persian word embeddings. b) How to get word embeddings of ParsBERT base model itself. c) How to get word embeddings of ParsGPT model.

### [Persian Word2Vec](https://github.com/AminMozhgani/Persian_Word2Vec)
A Persian Word2Vec Model trained by Wikipedia articles

### [Sentence Transformers (ParsBERT)](https://github.com/m3hrdadfi/sentence-transformers)
Three similar models based on fine-tuning ParsBERT base model on 3 different entailment datasets. Each of these models can be used for Semantic Search, Clustering, Summerization, Information retrieval and Topic Modeling tasks.

Benchmark
---------
### [ParsiNLU](https://github.com/persiannlp/parsinlu)
A comprehensive suite of high-level NLP tasks for Persian language. The dataset consists of the following tasks: Text entailment, Query paraphrasing, Reading comprehension, Multiple-choice QA, Machine translation and Sentiment analysis. They've been also fine-tuned [mt5](https://github.com/google-research/multilingual-t5) models on these datasets which result in various [Persian models](https://huggingface.co/persiannlp).

### [ParsBench - pb](https://github.com/shahriarshm/parsbench)
ParsBench provides toolkits for benchmarking LLMs based on the Persian language tasks.
- ParsiNLU all tasks
- Persian NER
- Persian Math
- ConjNLI Entailment
- Persian MMLU (khayyam Chanllenge)

### [Benchmarking ChatGPT for Persian](https://github.com/Ipouyall/Benchmarking_ChatGPT_for_Persian)
Benchmarking ChatGPT for Persian: A Preliminary Study
- Elemntry school
- Mathematical problems dataset

QA
--
### [PersianQA](https://github.com/sajjjadayobi/PersianQA)
Persian (Farsi) Question Answering Dataset. with models: [bert-base-fa-qa](https://huggingface.co/SajjadAyoubi/bert-base-fa-qa) with 162M parameters fine-tuned on this dataset and [xlm-roberta-large-fa-qa](https://huggingface.co/SajjadAyoubi/xlm-roberta-large-fa-qa) with 558M parameters fine-tuned on this dataset and SQuAD2.0 (English) dataset.

### [MeDiaPQA: A Question-Answering Dataset on Persian Medical Dialogues](https://data.mendeley.com/datasets/k7tzmrhr6n/1)
Medical Question Answering dataset consists of 15k dialogs in 70 specialities.

### [Persian-QA-Wikipedia](https://www.kaggle.com/datasets/amirpourmand/persian-qa-wikipedia)
26k QA and related excerpt extracted from Persian wikipedia. Some of the questions can not be answered based on the given excerpt by design (like SQuAD2.0).

### [ParsSQuAD](https://github.com/BigData-IsfahanUni/ParSQuAD)
Persian Question Answering Dataset based on Machine Translation of SQuAD 2.0

### [Crossword Cheat](https://github.com/dhpour/cwcheat)
Consists of 30K questions and answers of different Persian crosswords.

### [ParsiNLU](https://github.com/persiannlp/parsinlu)
Persian NLP team trained various mt5 and bert models on their multiple-choice QA dataset.

### [Persian Conversational Dataset (Legal)](https://huggingface.co/datasets/Kamtera/Persian-conversational-dataset)
It consists of 266k legal questions, answers and related tags.

### [Alpaca Persian](https://huggingface.co/datasets/sinarashidi/alpaca-persian)
Persian translation of 35k records of Stanford Alpaca Instruction dataset (52K records). There is also [a version with different formatting](https://huggingface.co/datasets/sinarashidi/alpaca-persian-llama2).

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

Entailment
----------
### [FarsTail: a Persian natural language inference dataset](https://github.com/dml-qom/FarsTail)
10k pairs with entailment label.

### [Sentence Transformers](https://github.com/m3hrdadfi/sentence-transformers)
Utilizes the FarsTail dataset for fine-tuning its [ParsBERT](https://github.com/hooshvare/parsbert) model, while also incorporating two other entailment datasets: [Wiki Triplet](https://drive.google.com/uc?id=1-lfrhHZwleYR4s0xGkXZPXxTeF25Q4C3) and [Wiki D/Similar](https://drive.google.com/uc?id=1P-KfNVIAx4HkaWFxc9aFoO3sHzHJFaVn).

### [ParsiNLU](https://github.com/persiannlp/parsinlu)
Persian NLP team trained various mt5 and bert models on their entailment dataset.

Datasets (classification)
-------------------------
### [Virgool Dataset](https://www.kaggle.com/datasets/mehranrafiee/persian-articles-for-labeling)
This could be a nice tool for Persian writers or bloggers to automatically pick the suggested hashtag or even subject for their articles. We could even collect data from google trend for each hashtag or 'label' used in an article. Consists of 11k+ articles.

### [BBC Persian Archive](https://www.kaggle.com/datasets/malekzadeharman/bbc-persian-archive)
The file contains 3780 news articles published by BBC Persian. The articles mostly belong to the year 1399 and 1400, and are published before Aban 18th, 1400. Columns are: title, publish_name, link, related_topics, body, category.

### [TasnimNews Dataset (Farsi - Persian) | تسنیم](https://www.kaggle.com/datasets/amirpourmand/tasnimdataset)
Consists of 63k News articles with following columns: category, title, abstract, body, time.

### [Farsnews-1398](https://www.kaggle.com/datasets/amirhossein76/farsnews1398)
Yearly collection of the Farsnews agency (1398). Contains 294k News article with following columns: title, abstract, paragraphs, cat, subcat, tags, link.

### [Digikala Magazine (DigiMag)](https://hooshvare.github.io/docs/datasets/tc#digikala-magazine-digimag)
A total of 8,515 articles scraped from Digikala Online Magazine. This dataset includes seven different classes: Video Games, Shopping Guide, Health Beauty, Science Technology, General, Art Cinema and Books Literature.

### [Miras Irony](https://github.com/miras-tech/MirasText/tree/master/MirasIrony)
Contains about 3K tweets, with each one of them labeled as either ironic or not.

### [Persian Stance Detection](https://github.com/Zarharan/PersianStanceDetection)
4K of records of stance detection in headlines and bodies of News articles.

### [A Stance datatset](https://github.com/sinarashidi/llama-2-persian)
Consists of 5.5K pairs of tweets which the stance of the reply tweets have been marked as against, support or neither to the main tweet.

### [A Claim datatset](https://github.com/sinarashidi/llama-2-persian)
Consists of 3.8K tweets which the type of each claims in each tweets have been identified. ~~But it does not show where is the claim located in the main tweet.~~

NER
---
### [Persian Twitter NER (ParsTwiner)](https://github.com/overfit-ir/parstwiner)
Name Entity Recognition (NER) on the Persian Twitter dataset. Consists of 6 entity types: event, location, natinality, organization and pog (political organizations and historical dynasties). ~~12k Named Entities in 232k tokens~~.

### [NSURL-2019 task 7: Named Entity Recognition (NER) in Farsi](https://github.com/nasrin-taghizadeh/NSURL-Persian-NER)
Extends [PEYMA corpus](https://arxiv.org/abs/1801.09936) (300k tokens), with another 600k tokens. Consists of 16 entity types including: date, location, percent number, money, time,  person and organization. ~~48k NEs in 884k tokens~~.

### [PersianNER (Arman)](https://github.com/HaniehP/PersianNER)
The dataset includes 250,015 tokens and 7,682 Persian sentences in total. Consists of 6 NE types including: facility, organization, location, event, person and proper noun. ~~37K NEs in 749k tokens~~.

### [Persian-NER](https://github.com/Text-Mining/Persian-NER)
Crowd-sourced NE dataset with 5 NE types. ~~2.2M NEs in 25M tokens.~~

### [ParsNER](https://github.com/hooshvare/parsner)
These dataset is a mixed NER dataset collected from [ARMAN](https://github.com/HaniehP/PersianNER), [PEYMA](https://arxiv.org/abs/1801.09936), and [WikiANN](https://github.com/afshinrahimi/mmner) that covered ten types of entities including: Date, Event, Facility, Location, Money, Organization, Percent, Person, Product and Time. 140K NEs in 40k sentences.

### [DaMuEL 1.0: A Large Multilingual Dataset for Entity Linking](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5047)

It is a large Multilingual Dataset for Entity Linking containing data in 53 languages including `Persian`. DaMuEL consists of two components: a knowledge base that contains language-agnostic information about entities, including their claims from Wikidata and named entity types (PER, ORG, LOC, EVENT, BRAND, WORK_OF_ART, MANUFACTURED); and Wikipedia texts with entity mentions linked to the knowledge base, along with language-specific text from Wikidata such as labels, aliases, and descriptions, stored separately for each language. [Paper](https://arxiv.org/pdf/2306.09288). For this project UDPipe has been used.

### [xtreme](https://github.com/google-research/xtreme)
XTREME is a benchmark for the evaluation of the cross-lingual generalization ability of pre-trained multilingual models that covers 40 typologically diverse languages and includes nine tasks. But for `Persian`
it only consists of:
- [Wikiann named entity recognition](https://github.com/afshinrahimi/mmner)
- Universal dependencies part-of-speech tagging (rasooli et al.)

Unlabled and Raw
--------

### [Persian SMS Dataset](https://www.kaggle.com/datasets/amirshnll/persian-sms-dataset)
Persian real SMS Dataset

### [Tarjoman (Persian Text) | ترجمان](https://www.kaggle.com/datasets/amirpourmand/tarjoman-persian-text)
Crawled more than 3k+ articles from tarjoman website.

### [Large-Scale Colloquial Persian](https://iasbs.ac.ir/~ansari/lscp/)
27M tweets. Although these texts have been labeled or translated using various NLP toolkits, they have never been supervised.

### [VOA 2003 - 2008](https://jon.dehdari.org/corpora/#persian)
Consists of 8M words with following columns: title, date, url and body.

### [Ensani-ir Abstrsacts](https://www.kaggle.com/datasets/amirpourmand/ensani-abstracts)
219K abstracts collected from Ensani.ir papers.

Toxic text
----------
### [Persian Abusive Words](https://github.com/mohamad-dehghani/Persian-Abusive-Words)
We created a dataset of 33338 Persian tweets, of which 10% contained Abusive words and 90% were non-Abusive.

### [Sansorchi](https://github.com/mmdbalkhi/Sansorchi)
Remove Persian (Farsi) Swear Words

### [Persian Swear Words](https://github.com/amirshnll/Persian-Swear-Words)
Persian Swear Dataset - you can use in your production to filter unwanted content. دیتاست کلمات نامناسب و بد فارسی برای فیلتر کردن متن ها

Stop word list
---------------

### [Persian stopwords collection](https://github.com/ziaa/Persian-stopwords-collection)
A collection of Persian stopwords. Consists of:
- [persian-stop-word](https://github.com/semnan-university-ai/persian-stop-word)
- [persian-stopwords](https://github.com/kharazi/persian-stopwords)
- and 5 other lists.

[All combined](https://github.com/rahmanidashti/PersianStopWords/)

### Different sources
- [Lucene](https://gist.github.com/dhpour/cd25e0affff5e921f7ebfb1e40dfe937)
- [Hazm](https://github.com/sobhe/hazm/blob/master/hazm/data/stopwords.dat)
- [awesome list](https://github.com/mhbashari/awesome-persian-nlp-ir/blob/master/stopwords.txt)
- [Dehghani](https://github.com/mohamad-dehghani/Semi-automatic-Detection-of-Persian-Stopwords-using-FastText-Library-/blob/main/pers_word.txt)
- [stopwords-fa](https://github.com/stopwords-iso/stopwords-fa/blob/master/stopwords-fa.txt)

### [Persian StopWords](https://www.kaggle.com/datasets/saeedtqp/sttoppword)
Consists of about 2k stop words.

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

Encyclopedia and Word Set
-------------------------

### [Vajehdan](https://github.com/sir-kokabi/Vajehdan)
Consists of following sets:
- Words of `Sareh` Dictionary (Purified Persian Words)
- `Farhangestan` chosen words for non-Persian equivalents.
- Farhange `Emlaee` (A dictionary of Persian orthography and spelling)
- A part of `Ganjoor`'s website poetry repos.
- Farhange `Motaradef` va Motazad (A dictionary of Persian synonyms and antonyms)
- Farhange `Teyfi` (Persian Thesaurus)

### [persian-names](https://github.com/nabidam/persian-names)
Persian names dataset

### [persian-names](https://github.com/armanyazdi/persian-names)
A Python package for generating random Persian (Farsi) names.

### [persian-wordlist](https://github.com/masterking32/persian-wordlist)
A SQL database that includes a dictionary of 494,286 Persian words.

### [persianwordjson](https://github.com/semnan-university-ai/persianwordjson)
This repository is a Persian meaningful database with json

### [persian-words-category](https://github.com/pfndesign/persian-words-category)
850k categorized Persian words.

### [similar-persian-words](https://github.com/pfndesign/similar-persian-words)
pre-calculated list of similar Persian words ordered by rating and best match

### [an-array-of-persian-words](https://github.com/pfndesign/an-array-of-persian-words)
List of ~240,000 Persian words

### [persian-databases](https://github.com/ganjoor/persian-databases)
Useful Persian dictionary and more. Consists of:
  - لغتنامه دهخدا [36097 ردیف]
  - مترادف های کلمات [19914 ردیف]
  - فرهنگ جامع عربی به فارسی [113342 ردیف]
  - فرهنگ جامع فارسی به عربی [32734 ردیف]
  - فرهنگ ابجد فارسی به عربی [41718 ردیف]
  - فرهنگ عربی به فارسی [8020 ردیف]
  - مفردات القرآن الکریم عربی [1609 ردیف]
  - مغاییس اللغه عربی [4669 ردیف]
  - المعجم الوسیط  عربی [41473 ردیف]
  - الامثال (شرح امثال عربی به عربی) [4510 ردیف]

### [Iranian job title](https://www.kaggle.com/datasets/amirshnll/iranian-job-title)
The "Iranian Job Title" dataset offers a comprehensive compilation of various job titles prevalent in Iran across diverse industries and sectors.

### [Moeen_thesaurus](https://github.com/kavir1698/Moeen_thesaurus)
فرهنگِ پیوندِ معناییِ واژه‌ها (بر پایه‌ی فرهنگ معین)

Poetry and Literature
---------------------
### [Hafez Poems](https://github.com/ArminGh02/hafez-poems-bot)
A simple Telegram bot implemented in Python.

### [Persian Databases](https://github.com/ganjoor/persian-databases)
Useful Persian dictionary and more. Consists of:
- اشعار شعرای ایرانی
    - احمد شاملو
    - باباطاهر
    - پروین اعتصامی
    - حافظ
    - خیام
    - رهی معیری
    - رودکی
    - سعدی
    - سهراب سپهری
    - شهریار
    - صائب تبریزی
    - عنصری
    - فردوسی
    - فروغ فرخزاد
    - مهدی اخوان ثالت
    - مولوی
    - نظامی
    - نیما یوشیج

- **دیتابیس قرآن کریم**
    - سوره های قرآن [114 ردیف]
    - آیات قرآن [6236 ردیف]
    - ترجمه الهی قمشه ای [6236 ردیف]
    - ترجمه کلمه به کلمه آیات [83668 ردیف]
    - لینک صوت قاریان [48 ردیف]

### [Shereno: A Dataset of Persian Modernist Poetry](https://www.kaggle.com/datasets/elhamaghakhani/persian-poems)
Collection of Persian Modernist Poetry from Iranian contemporary poets

### [Persian Poems Corpus](https://github.com/amnghd/Persian_poems_corpus)
Crawled Ganjoor for poems of 48 poets.

### [Persian Poet GPT2](https://huggingface.co/HooshvareLab/gpt2-fa-poetry)
This model fine-tuned on ParsGPT2 with [Chronological Persian poetry dataset](https://github.com/aghasemi/ChronologicalPersianPoetryDataset) and can generate poems by providing the name of the poet.

### [Chronological Persian Poetry Dataset](https://github.com/aghasemi/ChronologicalPersianPoetryDataset)
Dataset of poetry of 67 Persian poets of different times.

Audio
-------------
### [PSDR](https://github.com/Ralireza/PSDR)
Persian spoken digit recognition

### [Persian Questions](https://www.kaggle.com/datasets/halflingwizard/persian-questions)
Simple Persian Questions aimed to use in a voice assistant in 4 Categories. Labeled NEs in command utterances (in text).

### [Common Voice](https://github.com/common-voice/cv-dataset)
About 60 hours audio produced by various users reading sentences.
All sentences with duplicates are 500h+.

### [Persian Speech Corpus](https://fa.persianspeechcorpus.com/)
This ~2.5-hour Single-Speaker Speech corpus.

### [ShEMO: Persian Speech Emotion Detection Database](https://www.kaggle.com/datasets/mansourehk/shemo-persian-speech-emotion-detection-database)
A semi-natural db which contains emotional speech samples of Persian speakers. The database includes 3000 semi-natural utterances, equivalent to 3 h and 25 min of speech data extracted from online radio plays.

### [Speech to Text](https://github.com/shenasa-ai/speech2text)
A Deep-Learning-Based Persian Speech Recognition System. Takes advantage of various ASR platforms to create models for ASR. Also it uses various datasets including Mozzila CommonVoice and their own dataset which consists of 300h+ audio and transcription.

### [PCVC Speech Dataset](https://www.kaggle.com/datasets/sabermalek/pcvcspeech)
Phoneme based speech dataset.

### [Vosk](https://github.com/alphacep/vosk-api)
Open-source tool for speech recognition for various platforms and OSes, supprting 20 languages including `Persian`.

### [Wav2Vec2-Large-XLSR-53-Persian V3](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v3)
It is a wav2vec model fine-tuned on Mozzila CommonVoice Persian dataset. The model and the notebook to recreate the model with extra data are avaialble. 

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
Scripts and model developed for POS Tagging and Dependency Parsing Persian based on [TurboParser](http://www.ark.cs.cmu.edu/TurboParser).

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

Base Models
-----------

### [ParsBERT](https://github.com/hooshvare/parsbert)
Family of ParsBERT models including BERT, DistilBERT, ALBERT and ROBERTA. All of which are transformer based models with encoder-decoder design.

### [mBERT](https://huggingface.co/google-bert/bert-base-multilingual-cased)
Multilingual BERT model consists of 104 languages including `Persian`.

### [Shiraz](https://huggingface.co/lifeweb-ai/shiraz)
Is a BERT based model trained on Divan dataset (proprietary). This model has 46.6M parameters. Its evaluation on NER and Sentiment Analysis is repoted.

### [Tehran](https://huggingface.co/lifeweb-ai/tehran)
Is a BERT based model trained on Divan dataset (proprietary). This model has 124M parameters. Its evaluation on NER and Sentiment Analysis is repoted.

### [FaBERT](https://github.com/SBU-NLP-LAB/FaBERT)
Is a Persian BERT model trained on various Persian texts.

### [AriaBERT](https://huggingface.co/ViraIntelligentDataMining/AriaBERT)
Is a Persian BERT model trained on various Persian texts.

### [TookaBERT](https://huggingface.co/PartAI/TookaBERT-Base)
Is a Persian BERT model trained on various Persian texts with 123M parameters. There is also a large version of this [model](https://huggingface.co/PartAI/TookaBERT-Large) with 353M parameters.

Mocking
-------
### [PersianFaker](https://github.com/muhammadmp97/PersianFaker)
Do you need some fake data?

UI/UX
-----
### [Persian-Badge](https://github.com/kasraghoreyshi/persian-badge)
Persian-Badge is a website for having metadata badges in the Persian language

OCR
---

### [Handwritten city names in Arabic Persian](https://github.com/mehrdad-moradi/handwritten-city-names-in-Arabic-Persian)
This is a dataset of handwritten cities in Iran in Arabic/Persian that has been used in my Master project. This dataset is collected for sorting postal packages.

### [IranShahr](https://github.com/DarjaGFX/IranShahr/tree/master)
Hand-written / typed names of different cities of Iran in image format.

### [PLF Image Dataset](https://www.kaggle.com/datasets/sabermalek/plf50)
 50*50 Images of Persian letters (without dots) with 32 Different Fonts.

### [Persian Subwords](https://github.com/partdpai/persian_subwords)
Consists of about 20k images of Persian subwords in different fonts and sizes to be used in ocr models.

Spam
----

### [Persian SMS Spam Word](https://www.kaggle.com/datasets/amirshnll/persiansmsspamword)
persian sms spam word

Image Captioning
----------------

### [Coco 2017 Farsi](https://www.kaggle.com/datasets/navidkanaani/coco-2017-farsi)
Coco 2017 translated to Persian language.
91k images with caption in Persian.

### [Iranis dataset](https://github.com/alitourani/Iranis-dataset)
Dataset of Farsi License Plate Characters (83k).

### [ParsVQA-Caps](https://www.kaggle.com/datasets/maryamsadathashemi/parsvqacaps)
The VQA dataset consists of almost 11k images and 28.5k question and answer pairs with short and long answers usable for both classification and generation VQA.

### [CLIPfa](https://github.com/sajjjadayobi/CLIPfa)
A dataset consists of 16M records of images and their corresponding texts. It also consists of a model traind on 400k of this dataset for searching images based on text and image.

### [Persian Image Captioning](https://huggingface.co/datasets/SeyedAli/Persian-Image-Captioning)
Consists of about 26K records of images with th describing captions in Persian.

Translation
-----------

### [Persian movie dataset (English, Persian)](https://www.kaggle.com/datasets/mohammad26845/persian-movie-dataset-english-persian)
Persian language movies dataset from imvbox.com. 14k movies with storyline translated from Persian to English.

### [The Holy Quran](https://www.kaggle.com/datasets/zusmani/the-holy-quran)
Quran ayat with translation in 21 languages.

### [The Bible](https://github.com/christos-c/bible-corpus)
A multilingual parallel corpus created from translations of the Bible. In 100 languages including `Persian`.

### [W2C – Web to Corpus](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0022-6133-9)
A set of corpora for 120 languages including `Persian` automatically collected from wikipedia and the web.

### [ParsiNLU](https://github.com/persiannlp/parsinlu)
Persian NLP team trained various mt5 models on their translation dataset.

Knowledge Graph
---------------

### [PERLEX](http://farsbase.net/PERLEX.html)
2.7k Relation of entities with translation and relation type.

### [DaMuEL 1.0: A Large Multilingual Dataset for Entity Linking](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5047)

It is a large Multilingual Dataset for Entity Linking containing data in 53 languages including `Persian`. DaMuEL consists of two components: a knowledge base that contains language-agnostic information about entities, including their claims from Wikidata and named entity types (PER, ORG, LOC, EVENT, BRAND, WORK_OF_ART, MANUFACTURED); and Wikipedia texts with entity mentions linked to the knowledge base, along with language-specific text from Wikidata such as labels, aliases, and descriptions, stored separately for each language. [Paper](https://arxiv.org/pdf/2306.09288). For this project UDPipe has been used.

### [FarsBase](http://farsbase.net/about)
A knowledge graph platform for extracting from wikipedia, tables and unstructured texts. There is also a part of its data available for download.

### [Baaz](https://github.com/roshan-research/openie)
Open information extraction from Persian web.

### [ParsSimpleQA](https://github.com/partdpai/ParsSimpleQA)
The Persian Simple Question Answering Dataset and System over Knowledge Graph. It consists of 36k records.

### [ParsFEVER](https://github.com/Zarharan/ParsFEVER)
Is a Dataset for Farsi Fact Extraction and Verification based on [fever](https://github.com/awslabs/fever) guidelines.

Summary
-------
### [TasnimNews Dataset (Farsi - Persian) | تسنیم](https://www.kaggle.com/datasets/amirpourmand/tasnimdataset)
Consists of 63k News articles with following columns: category, title, `abstract`, body, time.

### [Farsnews-1398](https://www.kaggle.com/datasets/amirhossein76/farsnews1398)
Yearly collection of the Farsnews agency (1398). Contains 294k News article with following columns: title, `abstract`, paragraphs, cat, subcat, tags, link.

### [Wiki Summary](https://github.com/m3hrdadfi/wiki-summary)
95k documents with body and summery extracted from wikipedia Persian articles. There is also notebook to create and test models for summerization.

### [Persian Summarization](https://github.com/minasmz/Persian-Summarization)
Statistical and Semantical Text Summarizer in Persian Language

### [Persian News Summary](https://huggingface.co/datasets/HooshvareLab/pn_summary)
A well-structured summarization dataset for the Persian language consists of 93,207 records. It is prepared for Abstractive/Extractive tasks (like cnn_dailymail for English). It can also be used in other scopes like Text Generation, Title Generation, and News Category Classification.

### [Sentence Transformers (ParsBERT)](https://github.com/m3hrdadfi/sentence-transformers)
Consists of similar models fine-tuned on [ParsBERT](https://github.com/hooshvare/parsbert) using three different datasets, these models can be utilized for various applications, including Text summarization.

### [Miras Text](https://github.com/miras-tech/MirasText)
MirasText has more than 2.8 million articles and over 1.4 billion content words. Consists of following columns: content, summary, keywords, title, url.

Paraphrase
----------

### [ExaPPC](https://github.com/exaco/exappc)
Paraphrase data for Persian. It consists of 2.3M sentence pairs of which 1M of them are paraphrase and 1.3M are not parapharse of each other.

### [ParsiNLU](https://github.com/persiannlp/parsinlu)
Persian NLP team trained various mt5 models on their query paraphrase dataset.

### [Persian Text Paraphrase](https://huggingface.co/datasets/SeyedAli/Persian-Text-Paraphrasing)
Consists of 800 pairs of Persian sentences wich are paraphrases of each other.

WSD
---
### [SBU WSD Corpus](https://github.com/hrouhizadeh/SBU-WSD-Corpus)
SBU-WSD-Corpus: A Sense Annotated Corpus for Persian All-words Word Sense Disambiguation.

Generation
----------

### [Dorna Llama3 8B Instruct](https://huggingface.co/PartAI/Dorna-Llama3-8B-Instruct)
The Dorna models are a family of decoder-only models, specifically trained/fine-tuned on Persian data. This model is built using the Meta Llama 3 Instruct model. There are also [quantized versions](https://huggingface.co/PartAI/Dorna-Llama3-8B-Instruct-GGUF) of this model.

### [PersianLLaMA 13B Instruct](https://huggingface.co/ViraIntelligentDataMining/PersianLLaMA-13B-Instruct)
With 13 billion parameters, this model is fine-tuned using the [Persian Alpaca dataset](https://huggingface.co/datasets/sinarashidi/alpaca-persian) on Lllama 2 to excel at executing detailed instructions and delivering tailored outputs. There is also [PersianLLaMA 13B](https://huggingface.co/ViraIntelligentDataMining/PersianLLaMA-13B) which is fine-tuned on Persian wikipedia.

### [ParsGPT](https://github.com/hooshvare/parsgpt)
Persian version of GPT2 model fine-tuned on Persian poetry and [ParsiNLU sentiment analysis](https://github.com/persiannlp/parsinlu) datast.

### [AVA LLM collection](https://huggingface.co/collections/MehdiHosseiniMoghadam/ava-6648848e7a1ed3e0016f8395)
Fine-tuned versions of Mistral 7B and Llama 3 for Persian. The Persian resources used for these models are not known.

Thanks
------
Thanks to [Awesome Persian NLP](https://github.com/mhbashari/awesome-persian-nlp-ir) and [Awesome Iranian Datasets](https://github.com/MEgooneh/awesome-Iran-datasets) for providing some elements of this long list.