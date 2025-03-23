# Other

[![Check Links](https://github.com/dhpour/Persian-Text-Engineering-Hub/actions/workflows/link_check.yml/badge.svg)](https://github.com/dhpour/Persian-Text-Engineering-Hub/actions/workflows/link_check.yml)

Topics
------

- [Benchmark](#benchmark)
- [Entailment](#entailment)
- [Knowledge Graph](#knowledge-graph)
- [Embeddings](#embeddings)
- [Base Models](#base-models)

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

Entailment
----------
### [FarsTail: a Persian natural language inference dataset](https://github.com/dml-qom/FarsTail)
10k pairs with entailment label.

### [Sentence Transformers](https://github.com/m3hrdadfi/sentence-transformers)
Utilizes the FarsTail dataset for fine-tuning its [ParsBERT](https://github.com/hooshvare/parsbert) model, while also incorporating two other entailment datasets: [Wiki Triplet](https://drive.google.com/uc?id=1-lfrhHZwleYR4s0xGkXZPXxTeF25Q4C3) and [Wiki D/Similar](https://drive.google.com/uc?id=1P-KfNVIAx4HkaWFxc9aFoO3sHzHJFaVn).

### [ParsiNLU](https://github.com/persiannlp/parsinlu)
Persian NLP team trained various mt5 and BERT models on their entailment dataset.

Knowledge Graph
---------------

### [PERLEX](http://farsbase.net/PERLEX.html)
2.7k Relation of entities with translation and relation type.

### [DaMuEL 1.0: A Large Multilingual Dataset for Entity Linking](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5047)

It is a large Multilingual Dataset for Entity Linking containing data in 53 languages including `Persian`. DaMuEL consists of two components: a knowledge base that contains language-agnostic information about entities, including their claims from Wikidata and named entity types (PER, ORG, LOC, EVENT, BRAND, WORK_OF_ART, MANUFACTURED); and Wikipedia texts with entity mentions linked to the knowledge base, along with language-specific text from Wikidata such as labels, aliases, and descriptions, stored separately for each language. [Paper](https://arxiv.org/pdf/2306.09288). For this project UDPipe has been used.

### [FarsBase](http://farsbase.net/about)
It is a knowledge graph platform designed for extracting information from Wikipedia, tables, and unstructured texts. A portion of its data is also available for download.

### [Baaz](https://github.com/roshan-research/openie)
Open information extraction from Persian web.

### [ParsSimpleQA](https://github.com/partdpai/ParsSimpleQA)
The Persian Simple Question Answering Dataset and System over Knowledge Graph. It consists of 36k records.

### [ParsFEVER](https://github.com/Zarharan/ParsFEVER)
It is a dataset for Persian fact extraction and verification, developed in accordance with [FEVER](https://github.com/awslabs/fever) guidelines.

### [Entity Attribute Dataset 50k (GPT-4.0 Generated)](https://huggingface.co/datasets/BaSalam/entity-attribute-sft-dataset-GPT-4.0-generated-v1)
It includes detailed product information generated based on the title of each product, aiming to create a structured catalog in JSON format. The dataset encompasses a variety of product categories such as food, home and kitchen, clothing, handicrafts, tools, automotive equipment, and more.

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