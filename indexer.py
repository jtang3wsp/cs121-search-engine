import os
import json
import pickle
import time
import re
import math
from bs4 import BeautifulSoup
from urllib import parse
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer


# Takes in a beautifulsoup object and returns a list of tokens
def tokenize_page(content, ps):
    text = remove_css_and_script(content)
    tokens = word_tokenize(text)
    alnum_tokens = [ps.stem(token.lower()) for token in tokens if isalphanum(token) == True]
    bigram_tuple_list = list(ngrams(alnum_tokens, 2))
    bigram_str_list = []
    for bigram in bigram_tuple_list:
        bigram_str_list.append(bigram[0] + " " + bigram[1])
    alnum_tokens.extend(bigram_str_list) # indexes the bigrams as well
    return alnum_tokens


# Remove javascript and css code
# From https://stackoverflow.com/questions/30565404/remove-all-style-scripts-and-html-tags-from-an-html-page/30565420
def remove_css_and_script(soup):
    for script in soup(["script", "style"]): # remove all javascript and stylesheet code
        script.extract()
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


# Check if token is exclusively alphanumeric using ascii codes
def isalphanum(token):
    for c in token:
        if ord(c) <= 47 or (58 <= ord(c) <= 64) or (91 <= ord(c) <= 96) or ord(c) >= 123:
            return False
    return True


# Finds all tokens in bold or in headings, returns a set of tokens
def find_important_tokens(soup, ps):
    token_set = set()
    token_string = ""
    for header in soup.find_all(re.compile('^h[1-6]$')):
        token_string += header.text.strip() + " "
    for bold in soup.find_all('b'):
        token_string += bold.text.strip() + " "
    tokens = word_tokenize(token_string)
    alnum_tokens = [ps.stem(token.lower()) for token in tokens if isalphanum(token) == True]
    for token in alnum_tokens:
        token_set.add(token)
    return token_set


# Computes frequency for a token and adds to index
def build_index(index, counter, tokens, important_tokens):
    for token in tokens:
        if token not in index: # create a new token entry and add first posting
            index[token] = [[counter, 1, 0, 0]] 
        elif index[token][-1][0] != counter: # if the last posting is a different docID from current, create new posting
            index[token].append([counter, 1, 0, 0])
        else:
            index[token][-1][1] += 1
    for token in important_tokens:
        if token in index:
            index[token][-1][3] = 1 # set important field to 1
        else:
            index[token] = [[counter, 1, 0, 1]]
    return index


# Sorts and dumps the index to a text file
def sort_dump(index, file_num):
    sorted_index = dict(sorted(index.items())) # sorts dictionary by alpahebetical index
    index_file = open(f'index\\index{file_num}.txt', 'w')
    for token, posting in sorted_index.items():
        token_file_line = token  +  ":" + str(posting)
        index_file.write(token_file_line)
        index_file.write('\n')
    index_file.close()


# Saves a dictionary to a pickle file
def save_dict(filename, dict_to_save):
    with open(filename, 'wb') as f:
        pickle.dump(dict_to_save, f)


def strip_word(line: str) -> str:
    return line.split(":")[0]

def get_val(line: str) -> str:
    return line.split(":")[1][1:-1] # assuming [[X]]

# Merges all index files simultaneously
def merge_files_alphabetically(files_used: int):
    mega_token_file = open("index\\index.txt","w")
    file_list = [open(f"index\\index{x + 1}.txt") for x in range(files_used)]
    curr_file_line_list = [file.readline().strip() for file in file_list]
    word_list = [strip_word(line) for line in curr_file_line_list]
    smallest_word_index = [False for x in range(files_used)]
    smallest_word = word_list[0]
    smallest_word_index[0] = True
    
    while any(curr_file_line_list):
        # print(f'word options: {word_list}')
        # print(f'default smallest word: {smallest_word}')
        for word_index in range(1, len(word_list)):
            word = word_list[word_index]
            # print('word compared: ' + word)
            if smallest_word == "":
                smallest_word_index[word_index] = True
                smallest_word_index[word_index - 1] = False
                smallest_word = word
            if word != "" and word < smallest_word:
                # print(f'{word} comes before {smallest_word}')
                for x in range(word_index):
                    smallest_word_index[x] = False
                smallest_word = word
                smallest_word_index[word_index] = True
            elif word != "" and word == smallest_word:
                # print(f'{word} is found at {word_index} as well')
                smallest_word_index[word_index] = True

        # print(f'smallest word found: {smallest_word}')
        # print(smallest_word_index)

        if smallest_word_index.count(True) == 1:
            
            index_used = smallest_word_index.index(True)
            # print(f'only one found at index {index_used}')
            mega_token_file.write(curr_file_line_list[index_used] + '\n')
            
            # set up next comparisons
            curr_file_line_list[index_used] = file_list[index_used].readline().strip()
            word_list[index_used] = strip_word(curr_file_line_list[index_used])
            smallest_word_index[index_used] = False # revert back to false for next comparison
        
        else: # assuming more than one words is smallest
            #merge all values
            # print('merge values')
            merged_value = "["
            for x in range(files_used):
                if smallest_word_index[x]:
                    merged_value += get_val(curr_file_line_list[x]) + ", "
                    curr_file_line_list[x] = file_list[x].readline().strip()
                    word_list[x] = strip_word(curr_file_line_list[x])
                    smallest_word_index[x] = False
                    
            merged_value = merged_value[:-2] + "]\n"
            mega_token_file.write(smallest_word + ":" + merged_value)

                #apple:[[10, 5], [13, 7]]
        smallest_word = word_list[0]
        smallest_word_index[0] = True
        # print('------------------')

    for file in file_list:
        file.close()
    mega_token_file.close()

    # Delete
    for x in range(files_used):
        if os.path.exists(f"index\\index{x+1}.txt"):
            print(f"Deleting index{x+1}.txt...")
            os.remove(f"index\\index{x+1}.txt")


# Computes tf-idf for each posting       pass in counter-1 for doc_count
def compute_weights(filename, doc_count):
    weighted_index = open("index\\weighted_index.txt", "w")
    with open(filename, "r") as f:
        for line in f:
            token = line.split(":")[0]
            posting_list = eval(line.split(":")[1])
            doc_freq = len(posting_list) # Number of docs the token appears in
            idf = math.log(doc_count/doc_freq)
            for posting in posting_list:
                tf = 1 + math.log(posting[1])
                posting[2] = round(tf * idf, 4)
            weighted_index.write(f"{token}:{posting_list}\n")
    weighted_index.close()


# Indexes the index by byte offset
def index_of_index(index_file, doc_count):
    index_index = {}
    offset = 0
    with open(index_file, 'r') as f:
        for line in f:
            split = line.split(":")
            postings = eval(split[1])
            idf = math.log(doc_count/len(postings))
            index_index[split[0]] = [offset, len(postings), round(idf, 4)] # store info about the number of postings and idf
            offset += len(line) + 1
    return index_index


def main():
    directory = '.\\DEV\\'
    index = {}
    url_dict = {}
    url_set = set()
    counter = 1
    split = 10000
    file_num = 0
    ps = PorterStemmer()
    start_time = time.time()
    for subdir, dirs, files in os.walk(directory): # outer loop = each subdirectory in DEV
        for file in files: # inner loop = each file in a subdirectory
            with open(os.path.join(subdir, file), 'r') as f:
                print(f"{counter} ----- {os.path.join(subdir,file)}")
                data = json.load(f)
                url = parse.urldefrag(data['url'])[0]
                if url not in url_set:
                    soup = BeautifulSoup(data['content'], 'html5lib')
                    tokens = tokenize_page(soup, ps)
                    important_tokens = find_important_tokens(soup, ps)
                    index = build_index(index, counter, tokens, important_tokens)
                    url_dict[counter] = [url, file]
                    url_set.add(url)

                    if counter % split == 0:
                        # print(f"File count: {counter}")
                        sort_dump(index, counter // split)
                        file_num = counter // split
                        index.clear()
                    counter += 1
    
    if bool(index) == True: # Dump final index
        print("### DUMPING FINAL INDEX ###")
        sort_dump(index, counter//split+1)
        file_num = counter // split + 1

    print("### MERGING INDEX ###")
    merge_files_alphabetically(file_num) # merge indexes into one giant index
    print("### COMPUTING TF-IDF ###")
    compute_weights("index\\index.txt", counter-1) # compute tf-idf for each posting
    print("### INDEXING THE INDEX ###")
    pos_index = index_of_index("index\\weighted_index.txt", counter-1) # create index of the index

    # Save position index and url dict to txt and pickle files
    with open("index\\position_index.txt", "w") as f:
        for key, value in pos_index.items():
            f.write(f"{key}:{value}\n")
    save_dict("index\\pos_index.pk", pos_index)

    with open("index\\urls.txt", "w") as f:
        for key, value in url_dict.items():
            f.write(f"{key}:{value}\n")
    save_dict("index\\urls.pk", url_dict)

    with open("index\\counter.txt", "w") as f:
        f.write(str(counter-1))
    
    end_time = time.time()
    print(f"Total time: {end_time-start_time}")


if __name__ == "__main__":
    main()