import functions
import joblib


DEVELOPMENT_DOCS = 'data/devel.docs' 
DEVELOPMENT_QUERIES = 'data/devel.queries' 
DEVELOPMENT_QREL = 'data/devel.qrel' 
BITEXT_ENG = 'data/bitext.en' 
BITEXT_DE = 'data/bitext.de' 
NEWSTEST_ENG = 'data/newstest.en' 

'''documents = {}
f = open(DEVELOPMENT_DOCS, encoding='utf-8')

for line in f:
    doc = line.split("\t")
    terms = functions.extract_and_tokenize_terms(doc[1])
    documents[doc[0]] = terms
f.close()'''

documents = joblib.load('Documents')

from collections import defaultdict   
inverted_index = defaultdict(set)
for docid, terms in documents.items():
    for term in terms:
        inverted_index[term].add(docid) 

NO_DOCS = len(documents) 
AVG_LEN_DOC = sum([len(doc) for doc in documents.values()])/len(documents) 
tf_idf = joblib.load('TF-IDF')

f = open(BITEXT_ENG, encoding='utf-8')
train_sentences = []
for line in f:
    train_sentences.append(functions.tokenize(line))
f.close() 

unigram_counts, bigram_counts,trigram_counts = functions.get_counts(train_sentences)

token_count = sum(unigram_counts.values())
f = open(NEWSTEST_ENG)
test_sents = []
for line in f:
    test_sents.append(functions.tokenize(line))
f.close()


TRI_ONES = 0 
TRI_TOTAL = 0 
for twod in trigram_counts.values():
    for oned in twod.values():
        for val in oned.values():
            if val==1:
                TRI_ONES+=1 
            TRI_TOTAL += 1 
BI_ONES = 0 
BI_TOTAL = 0 
for oned in bigram_counts.values():
    for val in oned.values():
        if val==1:
            BI_ONES += 1 
        BI_TOTAL += 1 
UNI_ONES = list(unigram_counts.values()).count(1)
UNI_TOTAL = len(unigram_counts)

TRI_ALPHA = TRI_ONES/TRI_TOTAL #Alpha parameter for trigram counts
BI_ALPHA = BI_ONES/BI_TOTAL #Alpha parameter for bigram counts
UNI_ALPHA = UNI_ONES/UNI_TOTAL

'''from nltk.translate import IBMModel1
from nltk.translate import AlignedSent, Alignment
eng_sents = []
de_sents = []
f = open(BITEXT_ENG, encoding='utf-8')
for line in f:
    terms = functions.tokenize(line)
    eng_sents.append(terms)
f.close()
f = open(BITEXT_DE, encoding = 'utf-8')
for line in f:
    terms = functions.tokenize(line)
    de_sents.append(terms)
f.close()'''

eng_sents = joblib.load('EngSents')
de_sents = joblib.load('DeSents')

paral_sents = list(zip(eng_sents,de_sents))

from nltk.translate import IBMModel1
from nltk.translate import AlignedSent, Alignment

eng_de_bt = [AlignedSent(E,G) for E,G in paral_sents]
eng_de_m = IBMModel1(eng_de_bt, 5)

de_eng_bt = [AlignedSent(G,E) for E,G in paral_sents]
de_eng_m = IBMModel1(de_eng_bt, 5)

combined_align = []
for i in range(len(eng_de_bt)):
    forward = {x for x in eng_de_bt[i].alignment}
    back_reversed = {x[::-1] for x in de_eng_bt[i].alignment}   
    combined_align.append(forward.intersection(back_reversed))

de_eng_count = defaultdict(dict)
for i in range(len(de_eng_bt)):
    for item in combined_align[i]:
        de_eng_count[de_eng_bt[i].words[item[1]]][de_eng_bt[i].mots[item[0]]] =  de_eng_count[de_eng_bt[i].words[item[1]]].get(de_eng_bt[i].mots[item[0]],0) + 1
#Creating a English to German dict with occ count of word pairs
eng_de_count = defaultdict(dict)
for i in range(len(eng_de_bt)):
    for item in combined_align[i]:
        eng_de_count[eng_de_bt[i].words[item[0]]][eng_de_bt[i].mots[item[1]]] =  eng_de_count[eng_de_bt[i].words[item[0]]].get(eng_de_bt[i].mots[item[1]],0) + 1

de_eng_prob = defaultdict(dict)
for de in de_eng_count.keys():
    for eng in de_eng_count[de].keys():
        de_eng_prob[de][eng] = de_eng_count[de][eng]/sum(de_eng_count[de].values())
#Creating English to German dict with word translation probabilities 
eng_de_prob = defaultdict(dict)
for eng in eng_de_count.keys():
    for de in eng_de_count[eng].keys():
        eng_de_prob[eng][de] = eng_de_count[eng][de]/sum(eng_de_count[eng].values())
            
f = open(DEVELOPMENT_QUERIES, encoding='utf-8')
lno = 0
plno = 0
#Also building a dictionary of query ids and query content 
german_qs = {}
test_query_trans_sents = [] 
for line in f:
    lno+=1
    query_id = line.split('\t')[0]
    query_german = line.split('\t')[1]  
    german_qs[query_id] = query_german.strip()
    translation = str(functions.de_eng_noisy_translate(query_german,de_eng_prob,eng_de_prob,unigram_counts))
    if plno<5:
        print(query_id + "\n" + "German: " + str(query_german) + "\n" + "English: " + translation +"\n\n")
        plno+=1
    test_query_trans_sents.append(translation)
    if lno==100:
        break
f.close()

def transs(Query,rcount):
    translation = str(functions.de_eng_noisy_translate(Query,de_eng_prob,eng_de_prob,unigram_counts))
    return functions.retr_docs(translation,rcount,inverted_index,tf_idf)

def fetch_document_content(document_id):
    f = open(DEVELOPMENT_DOCS, encoding = 'utf-8')
    for line in f:
        doc = line.split("\t")
        if doc[0]==document_id:
            return doc[1]
            break
    f.close()