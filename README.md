# Passage-Retrieval-System
The repository shows a project finished in the course 'COMP0084 Information Retrieval and Data Mining' at UCL. 

## Project Introduction 

Techniques of information retrieval are applied widely on searching engines, question answering, and recommendations. The objective of this project is to construct an information retrieval system which solves the problem of retrieving passages for given queries based on information retrieval models including Cosine Similarity, BM25, and Query Likelihood Language Model (with Laplace smoothing, Lidstone smoothing, Dirichlet smoothing respectively).

The language of development in the coursework is Python 3.9.7. Due to the large volume of the dataset, the solution is written seperately in four '.py' files. The total runtime of four '.py' files is around 10 minutes on the First Generation Macbook Pro M1 (8-core CPU, 16GB RAM).

## Data sources
### 1. test-queries.tsv
A tab separated file, where each row contains a test query identifier (qid) and the actual query text.

### 2. passage-collection.txt
A collection of passages, one per row.

### 3. candidate-passages-top1000.tsv
A tab separated file with an initial selection of at most 1000 passages for each of the queries in test-queries.tsv. The format of this file is <qid pid query passage>, where pid is the identifier of the passage retrieved, query is the query text, and passage is the passage text (all tab separated). The passages contained in this file are the same as the ones in passage-collection.txt. Figure 1 shows some sample rows from that file.

![image](https://github.com/IvyZayn/Passage-Retrieval-System/blob/main/Image%20in%20README/sample%20rows.png)
<Figure 1: Sample rows from the candidate-passages-top1000.tsv file.>


## Deliveries

### 1. task1.py

#### 1) Objectives: 
To preprocess raw text data, get tokens, and qualitatively justify the dataset follows Zipf's law.

#### 2) Solutions: 
I define two function to get tokens by cleaning and steming texts and generate Zipf's law plots by plotting every token's normalised frequency against frequency ranking.

#### 3) Outcomes: 
a) The total number of 036 identified words is 107538.
b) Zipf's law plots are automatically saved. From the plot, I find the normalised frequency of a term is descending as its ranking increases with a shape similiar to the theoretical curve, which justifies that terms in the text set follow Zipf’s law qualitatively. Though the tendency of the empirical distribution is similar to the theoretical curve, some noices are still obvious on the log-log plot below, especially at large ranking values. The reason for the differences might be that the denominator becomes increasingly large while the ranking k and the sum of i increments but the numerator is constant (equal to 1). That leads to increasingly lower f. Besides, the limited size of the data set is also an element leading to noises on the curve.

 ![image](https://github.com/IvyZayn/Passage-Retrieval-System/blob/main/Output/Zipf'sLaw_plot.png)
<Figure 2: Term frequency ranking compared to Zipf's curve>

 ![image](https://github.com/IvyZayn/Passage-Retrieval-System/blob/main/Output/Zipf'sLaw_loglog.png)
<Figure 3: Term frequency ranking compared to Zipf's curve (log)>

c) The identified words and their frequencies are saved in 'passage_collection_stat.csv'.

### 2. task2.py

#### 1) Objectives: 
To get inverted index of every word in every passage for further retrieval.

#### 2) Solutions: 
To get an inverted index, I first use the function constructed in task1 to extract terms in the passages. 
For every passage I get a token, and all tokens are stored in a data structure of list-sublist. 
Then, I use the structure of ’word: passage: word frequency in the passage’ to store the inverted index, because in task3 I need to compute TF-IDF of passages which will use the frequency of a word in a particular passage.

#### 3) Outcomes: 
Generated inverted indices are saved in 'inverted_index.txt'.

### 3. task3.py

#### 1) Objectives: 
To retrieve 100 mostly relavant passages to every query by building retrieval models including Cosine Similarity and BM25. 

#### 2) Solutions: 
To build Cosine Similarity model, I compute TF-IDF vectors for passages and queries, based on which I compute Cosine Similarity scores between every query and passage. 
ALso, I build the BM25 model and compute the scores. 
Finally, I retrieve 100 mostly relavant relavant passages to every query based on the two models and saved the results respectively in two files.

#### 3) Outcomes: 
a) The cosine similarity scores between queries and 100 retrieved passages are stored in 'tfidf.csv' in the structure of <qid, pid, score>.
b) The BM25 scores between queries and 100 retrieved passages are stored in 'bm25.csv' in the structure of <qid, pid, score>.

### 4. task4.py

#### 1) Objectives: 
To retrieve 100 mostly relavant passages to every query by building query likelihood language models with Laplace smoothing, Lidstone correction, and Dirichlet smoothing.

#### 2) Solutions: 
I build query likelihood language models with Laplace smoothing, Lidstone correction, and Dirichlet smoothing, based on which I retrieve 100 mostly relavant passages to every query.

#### 3) Outcomes: 
The retrieval results based on query likelihood language models with Laplace smoothing, Lidstone correction, and Dirichlet smoothing are stored respectively in 'laplace.csv', 'lidstone.csv', 'dirichlet.csv' in the structure of <qid, pid, score>.
