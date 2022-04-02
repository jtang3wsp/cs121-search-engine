# cs121-search-engine
In console:
1. DEV folder is not included. Add the DEV file into this directory
2. Install all dependencies with pip (json, pickle, re, nltk, bs4, urllib)
3. For nltk, if it does not work, import nltk and add `python -m nltk.downloader stopwords` into cmd line or `nltk.download('stopwords')` into the main function 
2. Run indexer.py (Should take about 75 minutes) `python indexer.py`

To start the web interface:
(all in cmd prompt and cd into directory of search engine)
1. Download flask `pip install flask`
2. Run flask: `flask run`
3. Open a browser and it should run on this address: http://127.0.0.1:5000/

To type something in:
1. Type the query in the search bar input. 
2. Hit 'enter' or click the submit button and the links should appear
3. Query time would be found in cmd prompt
