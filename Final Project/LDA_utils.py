
from __future__ import division
import numpy as np

import string

import copy
# assume format is documents delimited by line 

def text_munge(filename):
    
    f = open(filename)
    text = f.read()
    text = text.lower()
    text = text.translate(string.maketrans("",""), string.punctuation)
    split_text = text.split('\r\n\r')

    #for i,doc in enumerate(split_text):
    #    d = doc.replace('\n','')
    #    d = d.replace('\r','')
    #    split_text[i]=d
        
    split_text = filter(None,split_text)

    docs=[]
    for paragraph in split_text:
        doc = paragraph.split(" ")
        docs.append(doc)


    uniqueWords=[]
    wordCounts=[]


    for doc in docs:
    
        for word in doc:
       
            if word in uniqueWords:
                for v,term in enumerate(uniqueWords):
                    if term==word:
                        wordCounts[v]+=1
                
            else:
                uniqueWords.append(word)
                wordCounts.append(1)
    
    V=len(uniqueWords) 
    dictionary = dict(zip(uniqueWords,range(0,V)))

    integer_docs=[]

    for doc in docs:
        integer_doc=[]
        for word in doc:
            integer_doc.append(dictionary[word])
        
        integer_docs.append(integer_doc)
 

    dictionary = {v: k for k, v in dictionary.items()}
    
    return dictionary,integer_docs,docs,wordCounts,uniqueWords



# Function name and purpose: LDA_generate - create a corpus generated according to latent Dirichlet allocation
#
# Parameters: M - number of documents in corpus, N - number of unique words in corpus, doc_length - number of words in each doc,
# alpha - vector used to generate Dirichlet distribution of topics, beta - vector used to generate Dirichlet distribution of words
# per topic, K - number of topics in corpus
#
# Returns: w_mn - matrix of word assignments such that each entry is a word in a document, z_mn - topic assignments for each word,
# theta_mk - probability of topic k occuring in document m, phi_kn - probability of word n occuring given topic k


def LDA_generate(V,M,K,doc_length,alpha,beta):
    assert len(alpha)==K
    assert len(beta)==V
    theta_mk = np.zeros([M,K])
    for i in range(0,M):
        theta_mk[i,:] = np.random.dirichlet(alpha,1)
    
    phi_kn = np.zeros([K,V])
    for k in range(0,K):
        phi_kn[k,:] = np.random.dirichlet(beta,1)
    
    
    z_mn = np.zeros([M,doc_length])
    w_mn = np.zeros([M,doc_length])
    for m in range(0,M):
        for n in range(0,doc_length):
            z_mn[m,n] = np.random.choice(K,p=theta_mk[m,:])
            w_mn[m,n] = np.random.choice(V,p=phi_kn[z_mn[m,n],:])

    return w_mn,theta_mk,phi_kn,z_mn
    
    
    