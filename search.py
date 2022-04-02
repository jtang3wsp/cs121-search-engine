from multiprocessing.reduction import duplicate
import time
import pickle
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from indexer import isalphanum
import nltk
index_file = open("index\\weighted_index.txt")
with open ("index\\counter.txt", "r") as f:
    doc_count = int(f.read())



# Loads a dictionary from a file
def load_index(filename):
    with open(filename, 'rb') as f:
        index = pickle.load(f)
    return index

pos_index = load_index("index\\pos_index.pk")
url_dict = load_index("index\\urls.pk")


def rank_document(query_result_dict: dict, top_k_dict: dict, num_tokens: int, posting: list, query_token_weight: int, query_normalized_length: float) -> int:
    docID = posting[0]
    doc_weight = posting[2]
    score = doc_weight * query_token_weight
    if docID not in query_result_dict: # if the document isn't scored yet
        query_result_dict[docID] = [score, math.pow(doc_weight, 2)]
        if posting[3] == 1: # if the token is important (bold/header)
            # print("IMPORTANT TOKEN")
            query_result_dict[docID][0] *= 20
    else: # add tf-idf score of current token to corresponding document
        query_result_dict[docID][1] += math.pow(doc_weight, 2)
        if posting[3] == 1:
            # print("IMPORTANT TOKEN")
            query_result_dict[docID][0] += score*20
        else:
            query_result_dict[docID][0] += score
    # Check if document is above threshold
    if query_result_dict[docID][0] > (75*math.log(num_tokens)+10):
        # print("Adding doc to top k")
        temp_score = query_result_dict[docID][0] #/ query_normalized_length
        if docID not in top_k_dict.keys():
            top_k_dict[docID] = temp_score
            return 1
        else:
            top_k_dict[docID] += temp_score
    return 0


# Searches a token in the index
def search(query_weight_dict: dict, index_file: object, pos_index: dict, query_result_dict: dict, top_k_dict: dict, counter: int, num_tokens: int, query_normalized_length: float):
    search_dict = {}
    for token in query_weight_dict:
        if token in pos_index:
            search_dict[token] = pos_index[token][1] # token: number of postings
        else:
            print(f"{token} not found")
    search_dict = dict(sorted(search_dict.items(), key=lambda item: item[1])) # sort by number of postings to search in ascending order
    # print(search_dict)
    duplicate_doc_list = []
    eval_time = 0
    for token in search_dict:
        #if token not in pos_index: # token is not indexed
            #print(f"{token} not found")
            #continue
        index_file.seek(pos_index[token][0])
        line = index_file.readline().rstrip()
        
        eval_time_start = time.time()
        posting_list_str = line.split(":")[1][0:-1] # gets rid of the [] at the outside
        posting_list_str_on_inside = posting_list_str.split("], [") # currently looks like [num, num, num], [num, num, num], [num, num, num] ==> ["[num, num, num", "num, num, num", "num, num, num]"]
        posting_list = [[float(val) for val in posting.strip("[").strip("]").split(", ")] for posting in posting_list_str_on_inside]
        # posting_list = eval(line.split(":")[1])
        eval_time_end = time.time()
        # print('posting list:' + str(posting_list))
        eval_time += eval_time_end-eval_time_start
        # print(f"Eval time: {eval_time}")
        
        # print(f"### Length of posting list for \"{token}\": {len(posting_list)}")
        # rank_time_start = time.time()
        for posting in posting_list:
            if posting[0] not in duplicate_doc_list:
                counter += rank_document(query_result_dict, top_k_dict, num_tokens, posting, query_weight_dict[token], query_normalized_length) # +1 if doc is added to top k dict
            if counter >= 10: # found 10 documents above score threshold
                # check for duplicate pages
                top_k_dict, counter = check_similar(top_k_dict, counter, duplicate_doc_list)
                # print(f"Counter: {counter}, len: {len(top_k_dict.keys())}")
                if counter >= 10:
                    # print("### FOUND TOP K ###")
                    # rank_time_end = time.time()
                    # print(f"Rank time: {rank_time_end-rank_time_start}")
                    return top_k_dict
        # rank_time_end = time.time()
        # print(f"Rank time: {rank_time_end-rank_time_start}")
    for docID in query_result_dict.keys():
        cos_score = query_result_dict[docID][0]
        angle = cos_score / (query_normalized_length*math.sqrt(query_result_dict[docID][1]))
        query_result_dict[docID] = cos_score
    return top_k_dict


def check_similar(top_k_dict, counter, duplicate_doc_list):
    top_k_dict = dict(sorted(top_k_dict.items(), key=lambda item: item[1], reverse=True))
    new_top_k_dict = {}
    prev_score = 0
    # print(len(top_k_dict.keys()))
    for key, value in top_k_dict.items():
        if -.1 < value-prev_score < .1:
            duplicate_doc_list.append(key)
            counter -= 1
        else:
            new_top_k_dict[key] = value
            prev_score = value
        # print(f"Counter: {counter}, len: {len(new_top_k_dict.keys())}")
    return new_top_k_dict, counter


def getSameLists(listA, listB):
    i = 0
    j = 0
    same_lists = list()
    while True:
        if len(listA) == i or len(listB) == j:
            break

        if listA[i] < listB[j]:
            i+=1
        elif listA[i] > listB[j]:
            j+=1
        else:
            same_lists.append(listA[i])
            i+=1
            j+=1
 
    return same_lists


# Prints top 10 results of the query
def print_results(result_dict, url_dict):
    # print(result_dict)
    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
    counter = 1
    for docID in result_dict:
        print(f"{counter}: Score = {result_dict[docID]} : {url_dict[docID][0]}   JSON file: {url_dict[docID][1]}")
        counter += 1
        if counter == 11:
            break


def generate_url(result_dict, url_dict):
    url = list()
    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
    counter = 1
    for docID in result_dict:
        url.append(url_dict[docID][0])
        counter += 1
        if counter == 11:
            break
    return url


# Computes tf-idf for query tokens {token: tf-idf}
def compute_weight(token_list, doc_count, pos_index):
    weight_dict = {}
    for token in token_list:
        if token in pos_index:
            if token not in weight_dict:
                weight_dict[token] = 1
            else:
                weight_dict[token] += 1
    for token in weight_dict:
        idf = pos_index[token][2]
        weight_dict[token] *= idf
    return weight_dict


def normalized_length(query_weight_dict):
    s = 0
    for token in query_weight_dict.keys():
        s += math.pow(query_weight_dict[token], 2)
    return math.sqrt(s)


def main():
    # index_file = open("index\\weighted_index.txt")
    with open ("index\\counter.txt", "r") as f:
        doc_count = int(f.read())
    pos_index = load_index("index\\pos_index.pk")
    url_dict = load_index("index\\urls.pk")
    ps = PorterStemmer()
    stop_words_set = set(stopwords.words('english'))
    while True:
        query = input("Enter your query (enter nothing to quit): ").lower()
        if query == "":
            break
        start_time = time.time()
        query_tokens = word_tokenize(query)
        # print(f"TOKENS: {query_tokens}")
        alnum_query_tokens = [ps.stem(token.lower()) for token in query_tokens if isalphanum(token) == True]
        # print(f"ALNUM TOKENS: {alnum_query_tokens}")
        num_tokens = len(alnum_query_tokens)
        if num_tokens == 0:
            print("No query terms found")
            continue
        # Add query bigrams
        bigram_tuple_list = list(ngrams(alnum_query_tokens, 2))
        bigram_str_list = []
        for bigram in bigram_tuple_list:
            bigram_str_list.append(bigram[0] + " " + bigram[1])
        alnum_query_tokens.extend(bigram_str_list)
        stop_word_count = 0
        for token in alnum_query_tokens: # count the number of stopwords in the query
            if token in stop_words_set:
                # print(f"### STOPWORD: {token}")
                stop_word_count += 1
        if stop_word_count/num_tokens < .8: # if the proportion of stopwords is below 80%, remove stopwords from query
            alnum_query_tokens = [token for token in alnum_query_tokens if token not in stop_words_set]
            num_tokens -= stop_word_count
        
        query_weight_dict = compute_weight(alnum_query_tokens, doc_count, pos_index)
        query_normalized_length = normalized_length(query_weight_dict)
        print(f"NUMBER OF TOKENS: {num_tokens} | Score threshold: {25*math.log(num_tokens)+10}")
        query_result_dict = {}
        top_k_dict = {}
        counter = 0
        top_k_dict = search(query_weight_dict, index_file, pos_index, query_result_dict, top_k_dict, counter, num_tokens, query_normalized_length)
        
            
        end_time = time.time()
        print(f"Query time: {end_time-start_time}")
        
        print(f"Your query \"{query}\" was found in these websites:")
        # print(len(top_k_dict.keys()))
        if len(top_k_dict.keys()) >= 10:
            #for docID in top_k_dict.keys():
                #top_k_dict[docID] /= math.sqrt(query_result_dict[docID][1])
            # print("GREATER THAN 10")
            print_results(top_k_dict, url_dict)
        else:
            print_results(query_result_dict, url_dict)
    
    index_file.close()


def one_input(query: str) -> list:
    
    ps = PorterStemmer()
    stop_words_set = set(stopwords.words('english'))
    if query == "":
        return list()
    start_time = time.time()
    query_tokens = word_tokenize(query)
    # print(f"TOKENS: {query_tokens}")
    alnum_query_tokens = [ps.stem(token.lower()) for token in query_tokens if isalphanum(token) == True]
    # print(f"ALNUM TOKENS: {alnum_query_tokens}")
    num_tokens = len(alnum_query_tokens)
    if num_tokens == 0:
        print("No query terms found")
        return list()
    # Add query bigrams
    bigram_tuple_list = list(ngrams(alnum_query_tokens, 2))
    bigram_str_list = []
    for bigram in bigram_tuple_list:
        bigram_str_list.append(bigram[0] + " " + bigram[1])
    alnum_query_tokens.extend(bigram_str_list)
    stop_word_count = 0
    for token in alnum_query_tokens: # count the number of stopwords in the query
        if token in stop_words_set:
            # print(f"### STOPWORD: {token}")
            stop_word_count += 1
    if stop_word_count/num_tokens < .8: # if the proportion of stopwords is below 80%, remove stopwords from query
        alnum_query_tokens = [token for token in alnum_query_tokens if token not in stop_words_set]
        num_tokens -= stop_word_count
    
    query_weight_dict = compute_weight(alnum_query_tokens, doc_count, pos_index)
    query_normalized_length = normalized_length(query_weight_dict)
    # print(f"NUMBER OF TOKENS: {num_tokens} | Score threshold: {25*math.log(num_tokens)+10}")
    query_result_dict = {}
    top_k_dict = {}
    counter = 0
    top_k_dict = search(query_weight_dict, index_file, pos_index, query_result_dict, top_k_dict, counter, num_tokens, query_normalized_length)
    
        
    end_time = time.time()
    print(f"Query time: {end_time-start_time}")

    # index_file.close()
    
    # print(f"Your query \"{query}\" was found in these websites:")
    # print(len(top_k_dict.keys()))
    if len(top_k_dict.keys()) >= 10:
        #for docID in top_k_dict.keys():
            #top_k_dict[docID] /= math.sqrt(query_result_dict[docID][1])
        # print("GREATER THAN 10")
        return generate_url(top_k_dict, url_dict)
    else:
        return generate_url(query_result_dict, url_dict)

if __name__ == "__main__":
    # nltk.download('stopwords')
    main()
