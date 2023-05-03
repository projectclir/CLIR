import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
import math

def tokenize(line, tokenizer=word_tokenize):
    utf_line = line.lower()
    return [token for token in tokenizer(utf_line)]

############################################################

import nltk
import re

stopwords = set(nltk.corpus.stopwords.words('english')) 
stemmer = nltk.stem.PorterStemmer() 

def extract_and_tokenize_terms(doc):
    terms = []
    for token in tokenize(doc):
        if token not in stopwords: 
            if not re.search(r'\d',token) and not re.search(r'[^A-Za-z-]',token): #Removing numbers and punctuations 
                terms.append(stemmer.stem(token.lower()))
    return terms

##################################################################

from collections import defaultdict 

def tf_idf_score(k1,b,term,docid,inverted_index,documents,NO_DOCS,AVG_LEN_DOC):  
    
    ft = len(inverted_index[term]) 
    term = stemmer.stem(term.lower())
    fdt =  documents[docid].count(term)
    
    idf_comp = math.log((NO_DOCS - ft + 0.5)/(ft+0.5))
    
    tf_comp = ((k1 + 1)*fdt)/(k1*((1-b) + b*(len(documents[docid])/AVG_LEN_DOC))+fdt)
    
    return idf_comp * tf_comp

def create_tf_idf(k1,b,inverted_index,documents,NO_DOCS, AVG_LEN_DOC):
    tf_idf = defaultdict(dict)
    for term in set(inverted_index.keys()):
        for docid in inverted_index[term]:
            tf_idf[term][docid] = tf_idf_score(k1,b,term,docid,inverted_index,documents,NO_DOCS,AVG_LEN_DOC)
    return tf_idf

###############################################################

def get_qtf_comp(k3,term,fqt):
    return ((k3+1)*fqt[term])/(k3 + fqt[term])


#Function to retrieve documents || Returns a set of documents and their relevance scores. 
def retr_docs(query,result_count,inverted_index,tf_idf):
    q_terms = [stemmer.stem(term.lower()) for term in query.split() if term not in stopwords] 
    fqt = {}
    for term in q_terms:
        fqt[term] = fqt.get(term,0) + 1
    
    scores = {}
    
    for word in fqt.keys():
        for document in inverted_index[word]:
            scores[document] = scores.get(document,0) + (tf_idf[word][document]*get_qtf_comp(1.5,word,fqt)) 
    
    return sorted(scores.items(),key = lambda x : x[1] , reverse=True)[:result_count] 

########################################################################

#Function to mark the first occurence of words as unknown, for training.
def check_for_unk_train(word,unigram_counts):
    if word in unigram_counts:
        return word
    else:
        unigram_counts[word] = 0
        return "UNK"

#Function to convert sentences for training the language model.    
def convert_sentence_train(sentence,unigram_counts):
    #<s1> and <s2> are sentinel tokens added to the start and end
    return ["<s1>"] + ["<s2>"] + [check_for_unk_train(token.lower(),unigram_counts) for token in sentence] + ["</s2>"]+ ["</s1>"]

#Function to obtain unigram, bigram and trigram counts.
def get_counts(sentences):
    trigram_counts = defaultdict(lambda: defaultdict(dict))
    bigram_counts = defaultdict(dict)
    unigram_counts = {}
    for sentence in sentences:
        sentence = convert_sentence_train(sentence, unigram_counts)
        for i in range(len(sentence) - 2):
            trigram_counts[sentence[i]][sentence[i+1]][sentence[i+2]] = trigram_counts[sentence[i]][sentence[i+1]].get(sentence[i+2],0) + 1
            bigram_counts[sentence[i]][sentence[i+1]] = bigram_counts[sentence[i]].get(sentence[i+1],0) + 1
            unigram_counts[sentence[i]] = unigram_counts.get(sentence[i],0) + 1
    unigram_counts["</s1>"] = unigram_counts["<s1>"]
    unigram_counts["</s2>"] = unigram_counts["<s2>"]
    bigram_counts["</s2>"]["</s1>"] = bigram_counts["<s1>"]["<s2>"]
    return unigram_counts, bigram_counts, trigram_counts

###########################################################################

'''token_count = sum(unigram_counts.values())'''

#Function to convert unknown words for testing. 
def check_for_unk_test(word,unigram_counts):
    if word in unigram_counts and unigram_counts[word] > 0:
        return word
    else:
        return "UNK"


def convert_sentence_test(sentence,unigram_counts):
    return ["<s1>"] + ["<s2>"] + [check_for_unk_test(word.lower(),unigram_counts) for word in sentence] + ["</s2>"]  + ["</s1>"]

#Returns the log probability of a unigram, with add-k smoothing.
def get_log_prob_addk(word,unigram_counts,k,token_count):
    return math.log((unigram_counts[word] + k)/ \
                    (token_count + k*len(unigram_counts)))

#Returns the log probability of a sentence.
def get_sent_log_prob_addk(sentence, unigram_counts,k,token_count):
    sentence = convert_sentence_test(sentence, unigram_counts)
    return sum([get_log_prob_addk(word, unigram_counts,k,token_count) for word in sentence])


def calculate_perplexity_uni(sentences,unigram_counts, token_count, k):
    total_log_prob = 0
    test_token_count = 0
    for sentence in sentences:
        test_token_count += len(sentence) + 2 # have to consider the end token
        total_log_prob += get_sent_log_prob_addk(sentence,unigram_counts,k,token_count)
    return math.exp(-total_log_prob/test_token_count)

###############################################################################

#Constructing trigram model with backoff smoothing

'''TRI_ALPHA = TRI_ONES/TRI_TOTAL #Alpha parameter for trigram counts
    
BI_ALPHA = BI_ONES/BI_TOTAL #Alpha parameter for bigram counts

UNI_ALPHA = UNI_ONES/UNI_TOTAL
    '''
def get_log_prob_back(sentence,i,unigram_counts,bigram_counts,trigram_counts,token_count,UNI_ALPHA,BI_ALPHA,TRI_ALPHA):
    if trigram_counts[sentence[i-2]][sentence[i-1]].get(sentence[i],0) > 0:
        return math.log((1-TRI_ALPHA)*trigram_counts[sentence[i-2]][sentence[i-1]].get(sentence[i])/bigram_counts[sentence[i-2]][sentence[i-1]])
    else:
        if bigram_counts[sentence[i-1]].get(sentence[i],0)>0:
            return math.log(TRI_ALPHA*((1-BI_ALPHA)*bigram_counts[sentence[i-1]][sentence[i]]/unigram_counts[sentence[i-1]]))
        else:
            return math.log(TRI_ALPHA*BI_ALPHA*(1-UNI_ALPHA)*((unigram_counts[sentence[i]]+0.0001)/(token_count+(0.0001)*len(unigram_counts)))) 
        
        
def get_sent_log_prob_back(sentence, unigram_counts, bigram_counts,trigram_counts, token_count,UNI_ALPHA,BI_ALPHA,TRI_ALPHA):
    sentence = convert_sentence_test(sentence, unigram_counts)
    return sum([get_log_prob_back(sentence,i, unigram_counts,bigram_counts,trigram_counts,token_count,UNI_ALPHA,BI_ALPHA,TRI_ALPHA) for i in range(2,len(sentence))])


def calculate_perplexity_tri(sentences,unigram_counts,bigram_counts,trigram_counts, token_count,UNI_ALPHA,BI_ALPHA,TRI_ALPHA):
    total_log_prob = 0
    test_token_count = 0
    for sentence in sentences:
        test_token_count += len(sentence) + 2 # have to consider the end token
        total_log_prob += get_sent_log_prob_back(sentence,unigram_counts,bigram_counts,trigram_counts,token_count,UNI_ALPHA,BI_ALPHA,TRI_ALPHA)
    return math.exp(-total_log_prob/test_token_count)

##########################################################################################

#Building noisy channel translation model
def de_eng_noisy(german,de_eng_prob,eng_de_prob,unigram_counts):
    noisy={}
    for eng in de_eng_prob[german].keys():
        noisy[eng] = eng_de_prob[eng][german]+ get_log_prob_addk(eng,unigram_counts,0.0001)
    return noisy

###########################################################################################

#Function for direct translation
def de_eng_direct(query,de_eng_prob,eng_de_prob):
    query_english = [] 
    query_tokens = tokenize(query)
    
    for token in query_tokens:
        try:
            query_english.append(max(de_eng_prob[token], key=de_eng_prob[token].get))
        except:
            query_english.append(token) 
    return " ".join(query_english)

#Function for noisy channel translation
def de_eng_noisy_translate(query,de_eng_prob,eng_de_prob,unigram_counts):  
    query_english = [] 
    query_tokens = tokenize(query)
    
    for token in query_tokens:
        try:
            query_english.append(max(de_eng_noisy(token,de_eng_prob,eng_de_prob,unigram_counts), key=de_eng_noisy(token,de_eng_prob,eng_de_prob,unigram_counts).get))
        except:
            query_english.append(token) 
    return " ".join(query_english)
            
#######################################################################################


