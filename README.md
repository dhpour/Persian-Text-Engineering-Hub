# Persian-Text-Engineering-Hub
[![Check Links](https://github.com/lingwndr/Persian-Text-Engineering-Hub/actions/workflows/link_check.yml/badge.svg)](https://github.com/lingwndr/Persian-Text-Engineering-Hub/actions/workflows/link_check.yml)

Curated list of Text libraries, tools and datasets for Persian language.

Topics
------
- [Multi-purpose libs](#multi-purpose-libs)
- [Grapheme to Phoneme](#graheme-to-phoneme)
- [Word Analyzing](#word-analyzing)
- [Sentiment Analysis](#sentiment-analysis)
- [Informal Persian](#informal-persian)
- [Numbers to Words](#numbers-to-words)
- [Word Embeddings](#word-embeddings)
- [Benchmark](#benchmark)
- [QA](#qa)
- [Dependency Parsing](#dependency-parsing)
- [Entailment](#entailment)
- [Datasets](#datasets)
- [NER](#ner)
- [Unlabled and Raw Text](#unlabled-and-raw)
- [Toxic Text](#toxic-text)
- [Stop Words List](#stop-words-list)
- [Spell Checking](#spell-checking)
- [Normalization](#normalization)
- [Transliteration](#transliteration)
- [Encyclopedia and Word Set](#encyclopedia-and-word-set)
- [Poetry and Literature](#poetry-and-literature)
- [Audio Dataset](#audio-dataset)
- [Crawl Suite](#crawl-suite)
- [POS Tag](#pos-tag)
- [Various](#various)
- [Mocking](#mocking)
- [UI/UX](#uiux)
- [OCR](#ocr)
- [Spam](#spam)
- [Image Captioning](#image-captioning)
- [Translation](#translation)
- [Knowledge Graph](#knowledge-graph)


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
- NER
- Morpheme Extracter
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

Word Analyzing
-------------
### [CPIA - Contemporary Persian Inflectional Analyzer](https://github.com/lingwndr/cpia)
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

### [PersianStemmingDataset](https://github.com/htaghizadeh/PersianStemmingDataset/tree/master)
Persian Stemming data-set in order to evaluate new stemmers

### [PersianStemmer-Python](https://github.com/htaghizadeh/PersianStemmer-Python)
PersianStemmer-Python

Sentiment Analysis
------------------
### [Persian-Sentiment-Resources](https://github.com/Text-Mining/Persian-Sentiment-Resources)
Awesome Persian Sentiment Analysis Resources - منابع مرتبط با تحلیل احساسات در زبان فارسی

- Consists of following datasets:
  - Deep Neural Networks in Persian Sentiment Analysis
  - Sentiment Analysis Challenges
  - Sentiment Lexicon
  - Sentiment Tagged Corpus (dataset)
  - HesNegar: Persian Sentiment WordNet

### [Persian-Sentiment-Analyzer](https://github.com/ashalogic/Persian-Sentiment-Analyzer)
Persian sentiment analysis ( آناکاوی سهش های فارسی | تحلیل احساسات فارسی )

### [Sentiment-Analysis](https://github.com/parsa-abbasi/Sentiment-Analysis)
Sentiment analysis using ML and DL models on Persian texts

### [LexiPers](https://github.com/phosseini/LexiPers)
A Sentiment Analysis Lexicon for Persian

### [Taaghche | طاقچه](https://www.kaggle.com/datasets/saeedtqp/taaghche)
Persian book comment ratings dataset. Consists of about 70k comment about 11k books.

### [Digikala (comments & products)](https://www.kaggle.com/datasets/radeai/digikala-comments-and-products)
The Digikala (comments & products) dataset offers a comprehensive glimpse into the vast online marketplace of Digikala, comprising over 1.2 million products and more than 6 million comments.

### [Digikala Comments](https://www.kaggle.com/datasets/soheiltehranipour/digikala-comments-persian-sentiment-analysis)
3k comments with score and ratings.

### [Persian tweets emotional dataset](https://www.kaggle.com/datasets/behdadkarimi/persian-tweets-emotional-dataset)
20k tweets with emotion identification labels.

### [Snappfood](https://hooshvare.github.io/docs/datasets/sa#snappfood)
Snappfood (an online food delivery company) user comments containing 70,000 comments with two labels (i.e. polarity classification): Happy, Sad.



Informal Persian
----------------

### [Shekasteh](https://github.com/rasoolims/Shekasteh)
Shekasteh is an evaluation dataset for Persian colloquial text. It comes from different genres, including blog posts, movie subtitles, and forum chats.

### [CPIA](https://github.com/lingwndr/cpia)
Informal and Formal Persian word analyzer (inflection with FST)

### [Persian Slang](https://github.com/semnan-university-ai/persian-slang)
Persian Slang Words (dataset)

Numbers to words
----------------

### [NumToPersian](https://github.com/Shahnazi2002/NumToPersian)
تبدیل عدد به حروف با پایتون

### [Convert numbers to Persian words](https://github.com/saeed-raeisi/num2words)
Read me this number python -- Convert number to Persian -- تبدیل عدد به فارسی

### [PersianNumberToWord](https://github.com/razavioo/PersianNumberToWord)
Convert numbers to Persian words.

### [dpern](https://github.com/amishbni/dpern)
Describe PERsian Numbers

Word Embeddings
---------------

### [Persian Word Embedding](https://github.com/miladfa7/Persian-Word-Embedding)
Persian Word Embedding using FastText, BERT, GPT and GloVe | تعبیه کلمات فارسی با روش های مختلف

### [Persian Word2Vec](https://github.com/AminMozhgani/Persian_Word2Vec)
A Persian Word2Vec Model trained by Wikipedia articles

Benchmark
---------
### [ParsiNLU](https://github.com/persiannlp/parsinlu)
A comprehensive suite of high-level NLP tasks for Persian language
- Text entailment
- Query paraphrasing
- Reading comprehension
- Multiple-choice QA
- Machine translation
- Sentiment analysis

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
Persian (Farsi) Question Answering Dataset (+ Models)

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

Entailment
----------
### [FarsTail: a Persian natural language inference dataset](https://github.com/dml-qom/FarsTail)
10k pairs with entailment label.

Datasets
--------
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

### [Persian Poems Corpus](https://github.com/amnghd/Persian_poems_corpus)
Crawled Ganjoor for poems 48 poets.

### [Large-Scale Colloquial Persian](https://iasbs.ac.ir/~ansari/lscp/)
27M tweets. Although these texts have been labeled or translated using various NLP toolkits, they have never been supervised.

Toxic text
----------
### [Persian Abusive Words](https://github.com/mohamad-dehghani/Persian-Abusive-Words)
We created a dataset of 33338 Persian tweets, of which 10% contained Abusive words and 90% were non-Abusive.

### [Sansorchi](https://github.com/mmdbalkhi/Sansorchi)
Remove Persian (Farsi) Swear Words

### [Persian Swear Words](https://github.com/amirshnll/Persian-Swear-Words)
Persian Swear Dataset - you can use in your production to filter unwanted content. دیتاست کلمات نامناسب و بد فارسی برای فیلتر کردن متن ها

Stop words list
---------------

### [PersianStopWords](https://github.com/rahmanidashti/PersianStopWords/)
A complete list of Persian stop words

### [persian-stop-word](https://github.com/semnan-university-ai/persian-stop-word)
persian stop word data

### [persian-stopwords](https://github.com/kharazi/persian-stopwords)
Persian (Farsi) Stop Words List

### [Persian-stopwords-collection](https://github.com/ziaa/Persian-stopwords-collection)
A collection of Persian stopwords - فهرست کلمات ایست فارسی
list other stop words collections (7 repos)

### [stopwords-fa](https://github.com/stopwords-iso/stopwords-fa)
Persian stopwords collection

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

### [Farsi-Normalizer](https://github.com/lingwndr/Farsi-Normalizer)
Simple Farsi normalizer

### [virastar](https://github.com/aziz/virastar)
Cleanning up Persian text! (Ruby)
- [Python version](https://github.com/JKhakpour/virastar.py)

### [Virastar (ویراستار)](https://github.com/brothersincode/virastar)
Virastar is a Persian text cleaner (JS).
- [PHP version 1](https://github.com/octoberfa/virastar)
- [PHP version 2](https://github.com/AlirezaSedghi/Virastar)
- [TypeScript version](https://github.com/nekofar/virastar)

Transliteration
---------------
### [Tajik-to-Persian transliteration](https://github.com/stibiumghost/tajik-to-persian-transliteration)
Tajik-to-Persian transliteration model

### [F2F](https://github.com/brothersincode/f2f)
Farsi to Finglish, a Persian transliterator

### [Behnevis](https://github.com/lingwndr/pinglish_behnevis)
24k ASCII transliterated Persian words

Encyclopedia and Word Set
-------------------------

### [Vajehdan](https://github.com/sir-kokabi/Vajehdan)
دستیار واژه‌گزینیِ فارسی.

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
A SQL database that includes a dictionary of 494,286 Persian words. دیکشنری 494,286 کلمه فارسی به صورت دیتابیس

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

Audio Dataset
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
A Deep-Learning-Based Persian Speech Recognition System. Takes advantage of various ASR platforms to create models for ASR. Also it uses various datasets including Mozzila CommonVoice and their own dtaset which consists of 300h+ audio and transcription.

### [PCVC Speech Dataset](https://www.kaggle.com/datasets/sabermalek/pcvcspeech)
Phoneme based speech dataset.

Crawl Suite
-----------
### [Persian News Search Engine](https://github.com/MehranTaghian/news-search-engine/tree/main)
A search engine for crawling news from the web, storing in a structured way, and querying through the stored documents for finding the most relevant results using Machine Learning and Information Retrieval techniques.

### [iranian-news-agencies-crawler](https://github.com/hamid/iranian-news-agencies-crawler)
a crawler to fetch last news from Iranian(Persian) news agencies.

### [PersianCrawler](https://github.com/pourmand1376/PersianCrawler)
Open source crawler for Persian websites including Asriran, fa-Wikipedia, Tasnim, Isna.

POS Tag
----------
### [Persian_POS_Tagger](https://github.com/AminMozhgani/Persian_POS_Tagger)
A Persian POS Tagger trained by The Persian Universal Dependency Treebank (Persian UD) with Tensorflow

### [PARSEME Corpse Fa](https://gitlab.com/parseme/parseme_corpus_fa)
PARSEME is a verbal multiword expressions (VMWEs) corpus for Farsi. All the annotated data come from a subset of the Farsi section of the [MULTEXT-East "1984"](https://nl.ijs.si/ME/Vault/V4/) annotated corpus 4.0. More than colums of LEMMA UPOS, XPOS, FEATS, HEAD and DEPREL there is also PARSEME:MVE which is manually annotated.

Various
------------------
### [Perke](https://github.com/AlirezaTheH/perke)
A keyphrase extractor for Persian

### [PREDICT-Persian-Reverse-Dictionary](https://github.com/arm-on/PREDICT-Persian-Reverse-Dictionary)
The first intelligent Persian reverse dictionary

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

Translation
-----------

### [Persian movie dataset (English, Persian)](https://www.kaggle.com/datasets/mohammad26845/persian-movie-dataset-english-persian)
Persian language movies dataset from imvbox.com. 14k movies with storyline translated from Persian to English.

### [The Holy Quran](https://www.kaggle.com/datasets/zusmani/the-holy-quran)
Quran ayat with translation in 21 languages.

Knowledge Graph
---------------

### [PERLEX](http://farsbase.net/PERLEX.html)
2.7k Relation of entities with translation and relation type.

### [DaMuEL 1.0: A Large Multilingual Dataset for Entity Linking](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5047)

It is a large Multilingual Dataset for Entity Linking containing data in 53 languages including `Persian`. DaMuEL consists of two components: a knowledge base that contains language-agnostic information about entities, including their claims from Wikidata and named entity types (PER, ORG, LOC, EVENT, BRAND, WORK_OF_ART, MANUFACTURED); and Wikipedia texts with entity mentions linked to the knowledge base, along with language-specific text from Wikidata such as labels, aliases, and descriptions, stored separately for each language. [Paper](https://arxiv.org/pdf/2306.09288). For this project UDPipe has been used.
