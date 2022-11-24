# Passage-Retrieval-System

## Project Introduction 

Techniques of information retrieval are applied widely on searching engines, question answering, and recommendations. The objective of this project is to construct an information retrieval system which solves the problem of retrieving passages for given queries based on Information Retrieval Models including Cosine Similarity, BM25, and Query Likelihood Language Model with Laplace smoothing, Lidstone smoothing, Dirichlet smoothing respectively.

The language of development in the project is Python 3.9.7. Due to the large volume of the dataset, the solution is written seperately in four '.py' files. The total runtime of four '.py' files is around 10 minutes on the First Generation Macbook Pro M1 (8-core CPU, 16GB RAM).

## Data sources
### 1. test-queries.tsv (200 rows)
A tab separated file, where each row contains a test query identifier (qid) and the actual query text.

### 2. passage-collection.txt (182,469 rows)
A collection of passages, one per row.

### 3. candidate-passages-top1000.tsv (189,877 rows)
A tab separated file with an initial selection of at most 1000 passages for each of the queries in 'test-queries.tsv'. The format of this file is 'qid pid query passage', where pid is the identifier of the passage retrieved, query is the query text, and passage is the passage text (all tab separated). The passages contained in this file are the same as the ones in passage-collection.txt. Figure 1 shows some sample rows from the file.

<div align=center>
<img src = "https://github.com/IvyZayn/Passage-Retrieval-System/blob/main/Image%20in%20README/sample%20rows.png" />
  
Figure 1: Sample rows from the 'candidate-passages-top1000.tsv' file.
</div>

## Deliveries

### 1. task1.py

#### 1) Objectives: 
To preprocess raw text data, get tokens, and qualitatively justify the dataset follows Zipf's law.

#### 2) Solutions: 
I define two functions to get tokens by cleaning and steming texts based on NLTK library and plot every token's normalised frequency against frequency ranking based on Zipf's law by matplotlib.

#### 3) Outcomes: 
a) The total number of identified words is 107538.
b) Zipf's law plots are automatically saved. From the plot below, I find the normalised frequency of a term is descending as its ranking increases with a shape similiar to the theoretical curve, which justifies that terms in the text set follow Zipf’s law qualitatively. 
 
<div align=center>
<img src = "https://github.com/IvyZayn/Passage-Retrieval-System/blob/main/Output/Zipf'sLaw_plot.png" />
  
Figure 2: Term frequency ranking compared to Zipf's curve
</div>
 
c) The identified words and their frequencies are saved in 'passage_collection_stat.csv'.


### 2. task2.py

#### 1) Objectives: 
To get inverted index of every word in every passage for further retrieval.

#### 2) Solutions: 
To get an inverted index, I first use the function constructed in task1 to extract terms in the passages. Then, I use the structure of ’word: passage: word frequency in the passage’ to store the inverted index.

#### 3) Outcomes: 
Generated inverted indices are saved in 'inverted_index.txt'.

### 3. task3.py

#### 1) Objectives: 
To retrieve 100 mostly relavant passages to every query by building retrieval models including Cosine Similarity and BM25. 

#### 2) Solutions: 
To build Cosine Similarity model, I compute TF-IDF vectors for passages and queries, based on which I compute Cosine Similarity scores between every query and passage. Also, I build the BM25 model and compute the scores. Finally, I retrieve 100 mostly relavant passages to every query based on the two models and saved the results respectively in two files.

#### 3) Outcomes: 
a) The Cosine Similarity scores between queries and 100 retrieved passages are stored in 'tfidf.csv' in the structure of <qid, pid, score>.
b) The BM25 scores between queries and 100 retrieved passages are stored in 'bm25.csv' in the structure of <qid, pid, score>.

### 4. task4.py

#### 1) Objectives: 
To retrieve 100 mostly relavant passages to every query by building query likelihood language models with Laplace smoothing, Lidstone correction, and Dirichlet smoothing.

#### 2) Solutions: 
I build query likelihood language models with Laplace smoothing, Lidstone correction, and Dirichlet smoothing, based on which I retrieve 100 mostly relavant passages to every query.

#### 3) Outcomes: 
The retrieval results based on query likelihood language models with Laplace smoothing, Lidstone correction, and Dirichlet smoothing are stored respectively in 'laplace.csv', 'lidstone.csv', 'dirichlet.csv' in the structure of <qid, pid, score>.
