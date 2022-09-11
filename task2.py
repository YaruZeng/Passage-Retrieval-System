import task1
import pandas as pd
from nltk.corpus import stopwords


def get_inverted_index(pid_list, candidate_passages): #generate inverted index
    
    # get tokens for candidate passages 
    passages_tokens = task1.extract_terms(candidate_passages)

    # set a dist to store words and pid-count pairs
    inverted_index = {}

    for line in passages_tokens:
        for word in line:
            inverted_index[word] = {}
          
    # get inverted index
    for ind_token in range(len(passages_tokens)): 
        pid = pid_list[ind_token]
        for word in passages_tokens[ind_token]:
            inverted_index[word][pid] = inverted_index[word].get(pid,0) + 1
    
    return inverted_index


def load_data(): # prepare data for inverted indexing

    # load identified words data generated from task1
    identified_words = pd.read_csv("passage_collection_stat.csv")
    identified_words = pd.DataFrame(identified_words)

    # remove stop words from identified words
    stop_words = stopwords.words('english')
    passage_collection_rsw = identified_words
    for item in stop_words:
        passage_collection_rsw = passage_collection_rsw[passage_collection_rsw["word"] != item]

    # load candidate_passages data
    candidate_passages = pd.read_csv("candidate-passages-top1000.tsv", sep='\t', header=None)
    candidate_passages.columns = ["qid","pid","query","passage"]

    # remove duplicated passages
    candidate_passages_distinct = candidate_passages.drop_duplicates(subset='pid',keep='first',inplace=False)

    # store data for generating inverted index
    pid_list = candidate_passages_distinct["pid"].tolist()
    candidate_passages = candidate_passages_distinct["passage"].tolist()


    return pid_list, candidate_passages


if __name__ == "__main__":

    pid_list, candidate_passages = load_data()

    inverted_index = get_inverted_index(pid_list, candidate_passages)

    # store inverted_index dist for further analysis
    s = str(inverted_index)
    f = open('inverted_index.txt','w')
    f.write(s)
    f.close()