# REConSum: the Requirements Elicitation Conversations Summarizer
## About
This code found in the `code` folder can be used to extract requirements-relevant questions from a transcription of a requirements elicitation session (an interview). It consists of three parts, plus an example. 

The first part (`preprocessing.py`) preprocesses our data, converting it to a format which can be used by the code. 
After that, the code in `find_questions.py` can identify the questions using either Part of Speech tags or Speech Acts. 
Finally, the code in `categorize_relevance.py` will determine whether these questions are requirements relevant or not, using TF-IDF.
These three Python files are orchestrated in the Jupyter notebook `example.ipynb`. 

The input for running the code can be added and found in the data folder.

The results of our tests can be found in the results folder.

## Requirements
To be able to run this code and replicate the example in the notebook, you will need to have the following:
- ([Python](https://www.python.org/downloads/) >= 3.7)
- ([Jupyter Notebook](https://jupyter.org/install) >= 6.4.0)

With the following packages:
- ([DialogTag](https://pypi.org/project/DialogTag/) >= 1.1.3)
- ([NLTK](https://www.nltk.org/install.html) >= 3.6)
- ([Numpy](https://pypi.org/project/numpy/) >= 1.20)
- ([Pandas](https://pypi.org/project/pandas/) >= 1.4.0)
- ([Pycorenlp](https://pypi.org/project/pycorenlp/) == 0.3.0)
- ([Sklearn](https://pypi.org/project/scikit-learn/) >= 1.0)
- ([Stemming](https://pypi.org/project/stemming/) == 1.0)
- ([Textract](https://pypi.org/project/textract/) >= 1.6.4)
- ([Tqdm](https://pypi.org/project/tqdm/) >= 4.62.0)

Furthermore, you will need some form of input; a transcription of a conversation. Our preprocessing code takes input generated by [AWS Transcribe](https://aws.amazon.com/transcribe/), otherwise adjustments to the preprocessing code have to be made.

Finally, to perform our TF-IDF comparison, we need the to use TFIDF terms from [a Wikipedia dump](https://github.com/SmartDataAnalytics/Wikipedia_TF_IDF_Dataset). Please place the wiki_tfidf_terms.csv in the data folder. 

In the example, we have put our input in a data folder and I would suggest doing the same:
<pre><code># For demonstration purposes, we will use one of our conversations from our dataset
transcription_file = "../data/example_conversation.txt"
wiki_tfidf_file = "../data/wiki_tfidf_terms.csv"
</code></pre>



## Running the code
In order to run the code, the requirements have to be met. Furthermore, if you wish to use Part of Speech tags to identify the questions, we need to run [StanfordCoreNLP](https://stanfordnlp.github.io/CoreNLP/download.html) locally.

Then, to listen to any calls made to `http://localhost:9000/`, we need to start it with the following command, in the directory in which StanfordCoreNLP was created:
<pre><code># Run the server using all jars in the current directory (e.g., the CoreNLP home directory)
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
</code></pre>


## Contact
The tool was developed by Xavier de Bondt (https://github.com/XavierdeBondt)
