import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from collections import Counter
import random

from nltk.corpus import stopwords
from collections import Counter
import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models import TfidfModel

import spacy
from nltk.corpus import stopwords

import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.gensim_models as gensimvis

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def lemmatization(comments, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    comments_out = []
    for comment in comments:
        doc = nlp(comment)
        new_comment = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_comment.append(token.lemma_)
        final = " ".join(new_comment)
        comments_out.append(final)
    return comments_out


def remove_stopwords(data_words, stopwords_list):
    return [[word for word in doc if word not in stopwords_list] for doc in data_words]


def gen_words(comments):
    final = []
    for comment in comments:
        new_comment = gensim.utils.simple_preprocess(comment, deacc=True)
        final.append(new_comment)
    return final


def make_bigrams(data_words, bigram):
    return [bigram[doc] for doc in data_words]


def make_trigrams(data_words, trigram):
    return [trigram[doc] for doc in data_words]


def create_bigram_trigram(data_words):
    bigrams_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=10)
    trigram_phrases = gensim.models.Phrases(bigrams_phrases[data_words], min_count=15, threshold=10)

    bigram = gensim.models.phrases.Phraser(bigrams_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = make_bigrams(data_words, bigram)
    data_bigrams_trigrams = make_trigrams(data_bigrams, trigram)
    return data_bigrams_trigrams


def compute_coherence_perplexity(corpus, texts, dictionary, params):
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=params['num_topics'],
                         random_state=0,
                         update_every=params['update_every'],
                         chunksize=params['chunksize'],
                         passes=params['passes'],
                         alpha=params['alpha'],
                         eta=params['eta'])
    perplexity = lda_model.log_perplexity(corpus)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model_lda.get_coherence()
    return coherence, perplexity, lda_model


# Function for random search
def random_search(param_grid, n_iter):
    param_combinations = list(itertools.product(*param_grid.values()))
    return random.sample(param_combinations, n_iter)


def perform_random_search():
    """
    # 1st Random Search - Online Learning
    # Best Params: {'num_topics': 10, 'update_every': 2, 'chunksize': 200, 'passes': 30, 'alpha': 'auto'} =	 Coherence: 0.5370863931633967
    param_grid = {
        'num_topics': [5, 7, 10],
        'update_every': [1, 2],
        'chunksize': [100, 200, 300],
        'passes': [10, 20, 30],
        'alpha': ['symmetric', 'auto']
    }
    """
    """
    # 2nd Random Search - Batch Learning
    # Best Params: {'num_topics': 10, 'update_every': 1, 'chunksize': 300, 'passes': 20, 'alpha': 'auto'} =	 Coherence: 0.5683360578973087
    param_grid = {
        'num_topics': [5, 7, 10],
        'update_every': [0],
        'passes': [10, 20, 30],
        'alpha': ['symmetric', 'auto']
    }
    """

    # 3rd Random Search - Online Learning
    # Added eta parameter
    # Best Params: {'num_topics': 10, 'update_every': 1, 'chunksize': 300, 'passes': 30, 'alpha': 'auto', 'eta': 'auto'} =	 Coherence: 0.5716926440164103
    """
    param_grid = {
        'num_topics': [5, 7, 10],
        'update_every': [1, 2],
        'chunksize': [200, 300],
        'passes': [20, 30],
        'alpha': ['auto'],
        'eta': ["auto", "symmetric"]
    }
    """
    """
    param_grid = {
        'num_topics': [5, 7, 10, 15],  # Try a range of topics
        'update_every': [1, 2],
        'chunksize': [100, 200, 300],
        'passes': [10, 20, 30],
        'alpha': ['symmetric', 'auto', 0.01, 0.1, 0.5],  # Experiment with different alpha values
        'eta': ['symmetric', 'auto', 0.01, 0.1, 0.5]  # Experiment with different beta values (eta in gensim)
    }
    """
    param_grid = {
        'num_topics': [5, 6, 7, 8],  # Try a range of topics
        'update_every': [1],
        'chunksize': [len(comments_df)],
        'passes': [40, 50],
        'alpha': [0.01, 0.05, 0.1],  # Experiment with different alpha values
        'eta': [0.01, 0.05, 0.1]  # Experiment with different beta values (eta in gensim)
    }

    n_iter = 35  # Number of random parameter combinations to try
    param_combinations = random_search(param_grid, n_iter)
    param_names = list(param_grid.keys())
    results = []

    best_model = None
    best_coherence = float('-inf')
    best_params = None

    for params in param_combinations:
        param_dict = dict(zip(param_names, params))
        coherence, perplexity, lda_model = compute_coherence_perplexity(corpus, data_bigrams_trigrams, id2word,
                                                                        param_dict)
        results.append((param_dict, coherence, perplexity))
        print(f'Params: {param_dict} => Coherence: {coherence}, Perplexity: {perplexity}')

        if coherence > best_coherence:
            best_coherence = coherence
            best_model = lda_model
            best_params = param_dict

    print(f'Best Params: {best_params} => Coherence: {best_coherence}')


def run_lda_model():
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=5,  # Adjust number of topics
                         update_every=1,  # More frequent updates
                         random_state=0,
                         passes=50,  # Increase passes
                         alpha=0.1,  # Adjust alpha value
                         eta=0.1,  # Adjust eta value
                         chunksize=500)  # Set chunksize to number of rows

    """(corpus=corpus,
                         id2word=id2word,
                         num_topics=12,
                         update_every=2,
                         random_state=0,
                         passes=20,
                         alpha=0.01,
                         chunksize=300,
                         eta=0.01
                         )
                         Perplexity: -13.261134481755496
Coherence Score: 0.456146487898412"""

    perplexity = lda_model.log_perplexity(corpus)
    print(f'Perplexity: {perplexity}')

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_bigrams_trigrams, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}')

    topics = lda_model.show_topics(num_topics=7, num_words=10, formatted=False)
    topic_dict = {f"Topic_{i + 1}": ", ".join([word for word, prob in words]) for i, (topic_num, words) in
                  enumerate(topics)}

    for topic_num, words in topics:
        print(f'Topic {topic_num}: {", ".join([word for word, prob in words])}')

    lda_vis_data = gensimvis.prepare(lda_model, corpus, id2word, mds="mmds", R=10)
    pyLDAvis.save_html(lda_vis_data, '../../outputs/nlp/topic_modeling/lda_topics_visualization.html')

    # Save topic probability distribution and max probability topic
    topic_assignments = [lda_model.get_document_topics(bow, minimum_probability=0) for bow in corpus]
    num_topics = lda_model.num_topics

    for doc_idx, topic_dist in enumerate(topic_assignments):
        topic_probabilities = np.zeros(num_topics)
        for topic_num, prob in topic_dist:
            topic_probabilities[topic_num] = prob

        for i in range(num_topics):
            comments_df.loc[doc_idx, f"Topic_{i + 1}_probability"] = topic_probabilities[i]

        max_topic_idx = topic_probabilities.argmax()
        comments_df.loc[doc_idx, 'Max_Probability_Topic'] = f"Topic_{max_topic_idx + 1}"

    # Map the topic words to the maximum probability topic
    comments_df['Max_Topic_Words'] = comments_df['Max_Probability_Topic'].apply(lambda x: topic_dict[x])

    topic_counts = comments_df['Max_Probability_Topic'].value_counts()
    print(topic_counts)
    comments_df.to_csv("../../outputs/nlp/topic_modeling/comments_with_lda_topics.csv")


if __name__ == '__main__':

    # add common words in the dataset that are too general to the stopwords
    stopwords_list = stopwords.words("english")
    stopwords_list.extend([
        'from', 'fly', 'flight', 'ryanair', 'travel',
        'like', 'just', 'get', 'got', 'would', 'one', 'also', 'could', 'us', 'said', 'go', 'going', 'see', 'even',
        'much', 'well', 'made', 'make', 'way', 'back', 'think', 'day', 'still', 'take', 'took', 'every', 'always',
        'really','many', 'say', 'done', 'know', 'look', 'looked', 'bit', 'lot', 'seems', 'seemed', 'etc', 'traveled', 'trip'])

    # Read the cleaned comments
    comments_df = pd.read_csv("../../outputs/nlp/sentiment_analysis/cleaned_comments.csv")
    if 'Unnamed: 0' in comments_df.columns:
        comments_df.drop(columns=["Unnamed: 0"], inplace=True)

    comments_df['lemmatized_Comment'] = lemmatization(comments_df['cleaned_Comment'])

    cleaned_comments = comments_df['lemmatized_Comment']

    # comments_df.to_csv("temp_lem.csv")
    # add an extra step of lemmatization
    # lemmatizated_comments = lemmatization(cleaned_comments)
    # print(lemmatizated_comments)

    data_words = gen_words(cleaned_comments)
    data_words = remove_stopwords(data_words, stopwords_list)

    data_bigrams_trigrams = create_bigram_trigram(data_words)

    id2word = corpora.Dictionary(data_bigrams_trigrams)

    # Checking the most frequent words to see if there is something to remove
    # word_frequencies = Counter({id2word[id]: freq for id, freq in id2word.dfs.items()})
    # most_common_words = word_frequencies.most_common(20)
    # for word, freq in most_common_words:
    #     print(f"{word}: {freq}")

    texts = data_bigrams_trigrams

    corpus = [id2word.doc2bow(text) for text in texts]

    tfidf = TfidfModel(corpus=corpus, id2word=id2word)

    low_value = 0.03
    words = []
    words_missing_in_tfidf = []

    for i in range(0, len(corpus)):
        bow = corpus[i]
        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words + words_missing_in_tfidf  # the words appearing in every comment are going to be removed
        for item in drops:
            words.append(id2word[item])
        words_missing_in_tfidf = [id for id in bow_ids if
                                  id not in tfidf_ids]  # The words with tf-idf score 0 will be missing
        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
        corpus[i] = new_bow

    # perform_random_search()
    run_lda_model()

    """
    perform_random_search()
    # Best Params: {'num_topics': 10, 'update_every': 1, 'chunksize': 300, 'passes': 30, 'alpha': 'auto', 'eta': 'auto'} =	 Coherence: 0.5716926440164103
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,
                                                update_every=1,
                                                random_state=0,
                                                passes=30,
                                                chunksize=300,
                                                alpha="auto"
                                                )


    with the best params the topic distribution among the comments are not optimal:
    Coherence Score: 0.5716926440164103
        Max_Probability_Topic
        Topic_2     877
        Topic_9     662
        Topic_6     430
        Topic_4     259
        Topic_7       8
        Topic_3       7
        Topic_8       4
        Topic_5       1
        Topic_10      1
        Name: count, dtype: int64
"""
