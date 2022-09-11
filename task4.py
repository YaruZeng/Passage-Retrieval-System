import task3
import pandas as pd
import math
import json


def load_data(): #load data for analysis

    test_queries = pd.read_csv("test-queries.tsv", sep='\t', header=None)
    test_queries.columns = ["qid","query text"]

    candidate_passages_top1000 = pd.read_csv("candidate-passages-top1000.tsv", sep='\t', header=None)
    candidate_passages_top1000.columns = ["qid","pid","query","passage"]

    f1 = open('inverted_index.txt','r')
    file1 = f1.read()
    inverted_index = eval(file1)
    f1.close()

    with open("f_queries.json") as file:
        f_queries_list = json.load(file)
        f_queries = dict(f_queries_list)

    with open("tf_passages.json") as file:
        tf_passages_list = json.load(file)
        tf_passages = dict(tf_passages_list)
    
    return test_queries, candidate_passages_top1000, inverted_index, f_queries, tf_passages


def smooth_compute(test_queries, candidate_passages_top1000, inverted_index, f_queries, tf_passages, qid_list): 
    # define a function to compute Laplace/Listone/Dirichlet smoothing

    tol_coll_word = 0
    miu = 50
    epsilon = 0.1
    V = len(inverted_index)

    Lap_smooth = {}
    Drl_smooth = {}
    Lids_smooth = {}

    # construct data structure
    for qid in qid_list:
        Lap_smooth[qid] = {}
        Lids_smooth[qid] = {}
        Drl_smooth[qid] = {}
    
    qid_candidate = {}
    for qid in qid_list:
        qid_candidate[qid] = list(candidate_passages_top1000["pid"][candidate_passages_top1000["qid"]==qid])

    # compute the total word occurrence in the collection
    cq_dict = {}
    for word, pid_count in inverted_index.items():
        word_occur = sum(pid_count.values())
        cq_dict[word] = word_occur
        tol_coll_word += word_occur 

    # compute Laplace/Listone/Dirichlet smoothing
    for qid, pid_word_f in f_queries.items():
        qid = int(qid)
        candidate_pid = qid_candidate[qid]
        
        for pid in candidate_pid:
            pid = int(pid)
            Lap_smooth[qid][pid] = 0
            Lids_smooth[qid][pid] = 0
            Drl_smooth[qid][pid] = 0
            
            for word, f in pid_word_f[str(pid)].items():
                cq = cq_dict[word]
                D = sum(tf_passages[str(pid)].values())
                
                Lap_smooth[qid][pid] += math.log((f+1)/(D+V))
                Lids_smooth[qid][pid] += math.log((f+epsilon)/(D+epsilon*V))
                
                item1 = (D/(D+miu))*(f/D)
                item2 = (miu/(D+miu))*(cq/tol_coll_word)
                Drl_smooth[qid][pid] += math.log(item1 + item2)

    return Lap_smooth, Lids_smooth, Drl_smooth


if __name__ == "__main__":

    test_queries, candidate_passages_top1000, inverted_index, f_queries, tf_passages  = load_data()
    qid_list = test_queries["qid"].tolist()

    Lap_smooth, Lids_smooth, Drl_smooth = smooth_compute(test_queries, candidate_passages_top1000, inverted_index, f_queries, tf_passages, qid_list)

    Lap_smooth_table = task3.output_data(Lap_smooth, qid_list)
    Drl_smooth_table = task3.output_data(Drl_smooth, qid_list)
    Lids_smooth_table = task3.output_data(Lids_smooth, qid_list)

    Lap_smooth_table.to_csv("laplace.csv", index = False, header = False) 
    Lids_smooth_table.to_csv("lidstone.csv", index = False, header = False)
    Drl_smooth_table.to_csv("dirichlet.csv", index = False, header = False)  