# Datasets

[![Check Links](https://github.com/dhpour/Persian-Text-Engineering-Hub/actions/workflows/link_check.yml/badge.svg)](https://github.com/dhpour/Persian-Text-Engineering-Hub/actions/workflows/link_check.yml)

Topics
---

- [QA](#qa)
- [Datasets (classification)](#datasets-classification)
- [NER](#ner)
- [Unlabled and Raw Text](#unlabled-and-raw)
- [Toxic Text](#toxic-text)
- [Stop Word List](#stop-word-list)
- [Encyclopedia and Word Set](#encyclopedia-and-word-set)
- [Poetry and Literature](#poetry-and-literature)
- [Audio Dataset](#audio)
- [OCR](#ocr)
- [Spam](#spam)
- [Image Captioning](#image-captioning)
- [Translation](#translation)
- [Summary](#summary)
- [Paraphrase](#paraphrase)
- [WSD](#wsd)

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
Consists of 30K questions and answers of various Persian crossword puzzles.

### [ParsiNLU](https://github.com/persiannlp/parsinlu)
Persian NLP team trained various mt5 and BERT models on their multiple-choice QA dataset.

### [Persian Conversational Dataset (Legal)](https://huggingface.co/datasets/Kamtera/Persian-conversational-dataset)
It consists of 266k legal questions, answers and related tags.

### [Alpaca Persian](https://huggingface.co/datasets/sinarashidi/alpaca-persian)
Persian translation of 35k records of Stanford Alpaca Instruction dataset (52K records). There is also [a version](https://huggingface.co/datasets/sinarashidi/alpaca-persian-llama2) with different formatting.

### [Synthetic Persian QA](https://huggingface.co/datasets/mshojaei77/Persian_QA)
This dataset contains 5900 Persian language question-answer pairs generated using the PersianAnswerGenerator class from answer.py. The answers are produced by an AI assistant leveraging the GPT-4o model through the Avala API service.

### [MauxiMix - Persian conversation dataset](https://huggingface.co/datasets/xmanii/mauxi-mix-persian)
MauxiMix is a carefully curated dataset of 1,000 high-quality Persian conversations, translated from the SmolTalk dataset using advanced language models. This dataset is specifically designed for training and fine-tuning Large Language Models (LLMs) with Supervised Fine-Tuning (SFT) techniques, contributing to the development of open-source Persian language models.

### [Crossword Puzzle Cheat Dataset](https://huggingface.co/datasets/PerSets/crossword-puzzle-persian-cheat)
This dataset consists of 30157 pairs of questions and answers.

### [Iranian Legal Question Answering Dataset](https://huggingface.co/datasets/PerSets/iran-legal-persian-qa)
This dataset includes over 570k questions and more than 1.9m answers, all in written form.

### [Clinical Question Answering Dataset II](https://huggingface.co/datasets/PerSets/clinical-persian-qa-ii)
This dataset contains more than 211k questions and more than 700k answers, all produced in written form.

### [Clinical Question Answering Dataset I](https://huggingface.co/datasets/PerSets/clinical-persian-qa-i)
This dataset contains approximately 50k questions and around 60k answers, all produced in written form.

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
Consists of 3.8K tweets, in which the type of each claim in each tweet have been identified. ~~But it does not show where is the claim located in the main tweet.~~

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
  - Dehkhoda dictionary (36k)
  - Synonyms (20k)
  - Arabic to Persian dictionary (113k)
  - Persian to Arabic dictionary(32k)
  - Abjad Persian to Arabic dictionary (42k)
  - Arabic to Persian dictionary (8k)
  - Quran Mofradat (1.6k)
  - Arabic monolingual dictionary (4.6k)
  - Intermediate Arabic dictionary (41k)
  - Alamsal - Arabic proverbs dictionary (4.5k)

### [Iranian job title](https://www.kaggle.com/datasets/amirshnll/iranian-job-title)
The "Iranian Job Title" dataset offers a comprehensive compilation of various job titles prevalent in Iran across diverse industries and sectors.

### [Moeen_thesaurus](https://github.com/kavir1698/Moeen_thesaurus)
Moeen dictionary based Thesaurus for Persian.

### [Enahnced Flexicon](https://github.com/asdoost/Enhanced_Flexicon)
It's an enhanced version of Flexicon word list with syllable, IPA procunciation and some refinements in word list itself.

Poetry and Literature
---------------------
### [Hafez Poems](https://github.com/ArminGh02/hafez-poems-bot)
A simple Telegram bot implemented in Python.

### [Persian Databases](https://github.com/ganjoor/persian-databases)
Useful Persian dictionary and more. Consists of:
- Persian poetry of Iranian poets:
  - Ahmad Shamlou
  - Baba-Taher
  - Parvin E'tesami
  - Hafez
  - Khayyam
  - Rahi-Moayeri
  - Roodaki
  - Sa'di
  - Sohrab Sepehri
  - Shahriar
  - Saeb Tabrizi
  - Onsori
  - Ferdowsi
  - Forugh Farrokhzad
  - Mehdi Akhavan Sales
  - Mowlavi
  - Nezami
  - Nima Yushij
- Quran Database
  - Quran Surahs (114)
  - Quran Versus (6236)
  - Quran Versus Translation by Gomshe'i (6326)
  - Quran Translation Word by word (83668)
  - Reading voice of Famous Readers (48)

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