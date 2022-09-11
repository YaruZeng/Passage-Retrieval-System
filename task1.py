import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.porter import *
import string

def extract_terms(file):

    porter = PorterStemmer()
    
    tokens = []
    extra = string.punctuation + "‘" + "’" + "“" + "”"
    for line in file:
        line = line.lower() # lower all characters
         # tokenisation (1-grams)
        for c in extra :
            line = line.replace(c, " ")# remove punctuations

        line_new = line.split()

        line_token = []
        for token in line_new:
            stemmed_token = porter.stem(token.strip())
            line_token.append(stemmed_token)

        tokens.append(line_token)
    
    return tokens


def zipf_plot(tokens):

    counts = {}
    for i in range(len(tokens)):
        for word in tokens[i]: # count occurrences of words 
            counts[word] = counts.get(word,0) + 1
    counts = list(counts.items())
    counts.sort(key=lambda x:x[1], reverse=True) # order words by occurrences

    print(f"identified index of terms: {len(counts)}")

    # create a dataframe to store data
    data = pd.DataFrame(counts, columns=["word","occurrence"])
    data.index = data.index + 1
    data["frequency"] = data["occurrence"]/tol_words
    data["rank"] = data.index
    data = data[["rank","word","occurrence","frequency"]]
    data.to_csv("passage_collection_stat.csv", index = False) # store data for further analysis

    # generate data for standard Zipf's curve
    N = max(data["rank"])
    k = np.arange(1, N)
    Hn = 0

    for i in range(1, N + 1):
        Hn += (1 / i)
    f = 1/ (k * Hn)
    
    # plot and save graphs
    plt.ion()
    plt.title("Zipf's Law Comparation")
    plt.xlabel("Term frequency ranking")
    plt.ylabel("Term prob. of occurrence")
    plt.plot(data["rank"],data["frequency"], color='blue', label = "data")
    plt.plot(k, f, linestyle="--", color='black', label="theory (Zipf's curve)")
    plt.legend()
    plt.savefig("Zipf'sLaw_plot.png")
    plt.ion()
    plt.show()

    plt.ion()
    plt.title("Zipf's Law Comparation")
    plt.xlabel("Term frequency ranking (log)")
    plt.ylabel("Term prob. of occurrence (log)")
    plt.loglog(data["rank"],data["frequency"], color='blue', label = "data")
    plt.loglog(k, f, linestyle="--", color='black', label="theory (Zipf's curve)")
    plt.savefig("Zipf'sLaw_loglog.png")

    plt.ion()
    plt.show()


if __name__ == "__main__":

    txt = open("passage-collection.txt")
    tokens = extract_terms(txt)
    tol_words = 0
    for line in tokens:
        for word in line:
            tol_words += 1

    zipf_plot(tokens)

