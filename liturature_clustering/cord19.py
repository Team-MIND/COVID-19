import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer as tf_idf
import numpy as np


# Path to general CORD-19 Data on my local machine
path = '/Users/ryersonburdick/Desktop/CORD-19-research-challenge'
# Path to biorxiv/medrxiv articles
biomed = path + '/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json'
# Path to comm use PMC articles
pmc = path + '/comm_use_subset/comm_use_subset/pmc_json'


class Doc:
    """
    Stores data from CORD-19 json files in more readable format.
    """

    def __init__(self, json_file):
        """
        Create Doc object from json file from CORD-19 dataset.

        json_file : str
            path to json file from CORD-19 datset from which to build Doc object
        """

        # Open json file
        doc = json.loads(open(json_file, "r").read())

        # Extract paper id
        self.id = doc['paper_id']

        # Extract paper title, make lowercase for normalization purposes
        self.title = doc['metadata']['title'].lower()

        # Extract author data (list of dicts contain first, last, etc.)
        _, self.authors = self.get_authors(doc)

        # Extract all text from abstract
        abstr = ''
        if 'abstract' in doc:
            self.abstr = ' '.join([par['text'] for par in doc['abstract']])
        # Extract all paragraphs from body
        body = ' '.join([par['text'] for par in doc['body_text']])
        self.text = abstr + ' ' + body

        # For storing feature vector
        self.vec = None


    def get_authors(self, doc):
        """
        Return authors dict and list of strings representing author names in
        more readable format.
        """

        authors = doc['metadata']['authors']
        author_strs = [
        author['first'] + ' ' + ' '.join(author['middle'])
        + ('' if author['middle'] == [] else ' ') + author['last']
        for author in authors]
        return authors, author_strs


def load_from_dirs(dirs):
    """
    Returns list of Doc objects from json files of all articles in specified directories.

    dir : list of str
        paths to directories containing CORD-19 json files.
    """

    doclist = []
    for dir in dirs:
        print("Reading files from directory: " + dir)
        for file in os.listdir(dir):
            if file.endswith('.json'):
                # Read file text
                fullpath = dir + '/' + file
                # Create Doc object and add to doclist
                doc = Doc(fullpath)
                doclist.append(doc)

    return doclist


def get_feature_vectors(doclist, vectorizer="tf-idf",
                        max_df=0.5):
    """
    Given a list of Doc objects, vectorizes each using selected
    document vectorization technique. Adds feature vector to Doc object
    and returns matrix of feature vectors.

    doclist : list of Doc
        List of articles to vectorize.
    vectorizer : {"tf-idf"}
        Document vectorization technique, default is TF-IDF.
    max_df : float
        Max document frequency for excluding common terms from feature vector.
        Default is 0.5.
    """

    corpus = [doc.text for doc in doclist]
    print("Performing " + vectorizer + " on " + str(len(doclist)) +
          " CORD-19 articles.")

    if vectorizer == "tf-idf":
        tf_idf_vectorizer = tf_idf(input='content',
                            strip_accents='unicode',
                            max_df=max_df)
        # Convert sklean.sparse.csr_matrix to numpy array
        result = tf_idf_vectorizer.fit_transform(corpus).toarray()

    # Store feature vectors in Doc objects
    for i in range(len(doclist)):
        doclist[i].vec = result[i]

    return result



def main():
    """
    Create sparse feature vectors for all biorxiv/medrxiv and pmc articles
    in CORD-19 dataset using TF-IDF.
    """

    doclist = load_from_dirs([biomed, pmc])

    # Return array of feature vectors and store feature vectors in doc objects
    mat = get_feature_vectors(doclist)

main()
