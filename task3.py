import task1
from collections import Counter
import pandas as pd
import math
import json


def load_data(): # load data for analysis

    test_queries = pd.read_csv("test-queries.tsv", sep='\t', header=None)
    test_queries.columns = ["qid","query text"]

    candidate_passages_top1000 = pd.read_csv("candidate-passages-top1000.tsv", sep='\t', header=None)
    candidate_passages_top1000.columns = ["qid","pid","query","passage"]

    f = open('inverted_index.txt','r') #inverted_index from task2
    file = f.read()
    inverted_index = eval(file)
    f.close()

    return test_queries, candidate_passages_top1000, inverted_index


def tfidf_passages(inverted_index): # compute TF-IDF for passages

    tf_passages = {}

    for word, pid_count in inverted_index.items():
        for pid, count in pid_count.items():
            tf_passages[pid] = {}
            
    for word, pid_count in inverted_index.items():
        for pid, count in pid_count.items(): 
            tf_passages[pid][word] = tf_passages[pid].get(pid,0) + inverted_index[word][pid]

    idf_passages = {}

    for word, pid_count in inverted_index.items():
        for pid, count in pid_count.items():
            idf_passages[pid] = {}
            
    N = len(idf_passages)

    for pid, word_count in tf_passages.items():
        for word, count in word_count.items():
            nt = len(inverted_index[word])
            idf_passages[pid][word] = math.log10(N/nt)

    tf_idf_passages = {}

    for word, pid_count in inverted_index.items():
        for pid, count in pid_count.items():
            tf_idf_passages[pid] = {}

    for pid, word_tf in tf_passages.items():
        for word, tf in word_tf.items():
            tf_idf_passages[pid][word] = tf_passages[pid][word]*idf_passages[pid][word]

    # output data for later analysis
    with open("tf_passages.json", "w", newline='', encoding="UTF-8") as file:
        json.dump(tf_passages, file)

    return tf_passages, idf_passages, tf_idf_passages


def tfidf_queries(test_queries): # compute TF-IDF for queries

    qid_list = test_queries["qid"].tolist()
    query_text = test_queries["query text"].tolist()
    query_text_tokens = task1.extract_terms(query_text)

    tf_queries = {}

    for qid in qid_list:
        tf_queries[qid] = {}

    for ind_token in range(len(query_text_tokens)):
        word_count = dict(Counter(query_text_tokens[ind_token]))

        qid = qid_list[ind_token]
        tf_queries[qid] = word_count

    idf_queries = {}

    for qid in qid_list:
        idf_queries[qid] = {}
            
    idf_words = {}

    for pid, word_idf in idf_passages.items():
        for word, idf in word_idf.items():
            if word not in idf_words.keys():
                idf_words[word] = idf

    for qid, word_count in tf_queries.items():
        for word in word_count.keys():
            if word in idf_words.keys():
                idf_queries[qid][word] = idf_words[word]
            else:
                idf_queries[qid][word] = 0

    tf_idf_queries = {}

    for qid in qid_list:
        tf_idf_queries[qid] = {}
        
    for qid, word_tf in tf_queries.items():
        for word, tf in word_tf.items():
            tf_idf_queries[qid][word] = tf_queries[qid][word]*idf_queries[qid][word]


    return tf_queries, idf_queries, tf_idf_queries, qid_list


def length_x(x_list): # define a function to compute the length of x
    
    length = 0
    sum_quare = 0
    for x in x_list:
        sum_quare += x*x
    length = pow(sum_quare,0.5)
    
    return length


def cos_similarity(qid_list,qid_candidate): # define a function to compute cosine similarity

    tfidf_len_queries = {}
    tfidf_len_passages = {}
    inner_product = {}
    cos_sim = {}
    
    # compute queries' length
    for qid, word_tfidf in tf_idf_queries.items():
        tfidf_len_queries[qid] = length_x(word_tfidf.values())

    # compute passagaes' length
    for pid, word_tfidf in tf_idf_passages.items():
        tfidf_len_passages[pid] = length_x(word_tfidf.values())

    # compute the inner product on the numenator
    for qid in qid_list:
        inner_product[qid] = {}

    for qid, word_tfidf_q in tf_idf_queries.items():
        for pid, word_tfidf_p in tf_idf_passages.items():
            word_list = set(word_tfidf_q.keys()).intersection(set(word_tfidf_p.keys()))
            inner_product[qid][pid] = 0
            for word in word_list:
                inner_product[qid][pid] += tf_idf_passages[pid][word]*tf_idf_queries[qid][word]

    # compute the cosine similarity score
    for qid in qid_list:
        cos_sim[qid] = {}
        
    for qid, pid_inner_product in inner_product.items():
        candidate_pid = qid_candidate[qid]
        for pid in candidate_pid:
            cos_sim[qid][pid] = inner_product[qid][pid]/(tfidf_len_queries[qid]*tfidf_len_passages[pid])

    return cos_sim


def bm(tf_passages, qid_candidate): # define a function to compute BM25

    N = len(tf_passages)
    k1 = 1.2
    k2 = 100
    b = 0.75
    
    n_queries = {}
    f_queries = {}
    BM = {}

    # construct the data structure
    for qid in qid_list:
        n_queries[qid] = {}
        f_queries[qid] = {}
        BM[qid] = {}

    # coumpute dl and avdl
    dl = {}
    for pid, word_count in tf_passages.items():
        dl[pid] = sum(word_count.values())

    avdl = sum(dl.values())/len(dl)

    # compute num of documents a word occurs in
    for qid, word_tf in tf_queries.items():
        for word in word_tf.keys(): 
            if word in inverted_index.keys():
                n_queries[qid][word] = len(inverted_index[word])
            else:
                n_queries[qid][word] = 0

    # compute num of documents a word occurs in
    for qid, word_tf in tf_queries.items():
        word_list = word_tf.keys()
        for pid, word_count in tf_passages.items():
            f_queries[qid][pid] = {}
            for word in word_list:
                if word in word_count.keys():
                    f_queries[qid][pid][word] = tf_passages[pid][word]

    # compute the frequency a word of queries occur in a document
    for qid, pid_dict in f_queries.items():
        candidate_pid = qid_candidate[qid]
        for pid in candidate_pid:
            BM[qid][pid] = 0
            for word, f in pid_dict[pid].items():
                n = n_queries[qid][word]
                qf = tf_queries[qid][word]
                K = k1*((1-b)+b*dl[pid]/avdl) 

                item1 = (N-n+0.5)/(n+0.5)
                item2 = ((k1+1)*f)/(K+f)
                item3 = ((k2+1)*qf)/(k2+qf)

                BM[qid][pid] += math.log(item1+item2+item3)

    # output data for later analysis
    with open("f_queries.json", "w", newline='', encoding="UTF-8") as file:
        json.dump(f_queries, file)

    return BM
    

def output_data(data, qid_list): # define a function to output data

    qid_pid_score = pd.DataFrame() # create a dataframe to store data

    for qid in qid_list:
        pid_score = sorted(data[qid].items(),key=lambda x:x[1], reverse=True) # order by score reversely
        pid_score_table = pd.DataFrame(pid_score,columns = ["pid","score"])
        pid_score_table["qid"] = qid
        
        if len(pid_score_table)>=100:
            pid_score_table = pid_score_table.iloc[0:100,]
        
        qid_pid_score = pd.concat([qid_pid_score,pid_score_table])
        
    qid_pid_score = qid_pid_score[["qid","pid","score"]]
    
    return qid_pid_score


if __name__ == "__main__":

    # retrieve by cosine similarity
    test_queries, candidate_passages_top1000, inverted_index = load_data()
    tf_passages, idf_passages, tf_idf_passages = tfidf_passages(inverted_index)
    tf_queries, idf_queries, tf_idf_queries, qid_list = tfidf_queries(test_queries)

    qid_candidate = {}
    for qid in qid_list:
        qid_candidate[qid] = list(candidate_passages_top1000["pid"][candidate_passages_top1000["qid"]==qid])

    cos_sim = cos_similarity(qid_list, qid_candidate)

    # retrieve by BM25
    BM = bm(tf_passages, qid_candidate)
    
    # output data
    cos_sim_table = output_data(cos_sim, qid_list)
    cos_sim_table.to_csv("tfidf.csv", index = False, header = False)

    bm_table = output_data(BM, qid_list)
    bm_table.to_csv("bm25.csv", index = False, header = False)













