import pandas as pd
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
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


def topic_modeling():
    """
    Main function to perform topic modeling on comments.
    """
    # Read the preprocessed comments for sentiment analysis
    comments_df = pd.read_csv("../../outputs/nlp/sentiment_analysis/cleaned_comments.csv")
    if 'Unnamed: 0' in comments_df.columns:
        comments_df.drop(columns=["Unnamed: 0"], inplace=True)

    # Lemmatize comments
    comments_df['lemmatized_Comment'] = lemmatization(comments_df['cleaned_Comment'])

    cleaned_comments = comments_df['lemmatized_Comment']

    data_words = gen_words(cleaned_comments)
    data_words = remove_stopwords(data_words)

    data_bigrams_trigrams = make_bigram_trigram(data_words)

    id2word = corpora.Dictionary(data_bigrams_trigrams)

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

    """
    Do hyperparameter optimization with a random search implementation.
    Since RandomizedSearchCV is optimized for the scikit library and gensim is used in this implementation a random search is perform with the following functionality.
    Uncomment the following line to perform random search.
    """
    # perform_random_search()
    run_lda_model(comments_df, corpus, id2word, data_bigrams_trigrams)


def lemmatization(comments, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """
    Lemmatizes the comments using spaCy.

    Parameters:
        comments (list): List of comments to lemmatize.
        allowed_postags (list): List of part-of-speech tags to consider for lemmatization.

    Returns:
        list: List of lemmatized comments.
    """
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


def remove_stopwords(data_words):
    """
    Removes stopwords from the data.

    Parameters:
        data_words (list): List of tokenized words.

    Returns:
        list: List of words with stopwords removed.
    """
    # Create the stopwords list
    stopwords_list = stopwords.words("english")
    # Extend with dataset specific keywords that are too general for topic modeling
    stopwords_list.extend([
        'from', 'fly', 'flight', 'ryanair', 'travel',
        'like', 'just', 'get', 'got', 'would', 'one', 'also', 'could', 'us', 'said', 'go', 'going', 'see', 'even',
        'much', 'well', 'made', 'make', 'way', 'back', 'think', 'day', 'still', 'take', 'took', 'every', 'always',
        'really', 'many', 'say', 'done', 'know', 'look', 'looked', 'bit', 'lot', 'seems', 'seemed', 'etc', 'traveled',
        'trip'])
    return [[word for word in doc if word not in stopwords_list] for doc in data_words]


def gen_words(comments):
    """
    Generates words from comments.

    Parameters:
        comments (list): List of comments.

    Returns:
        list: List of tokenized words.
    """
    final = []
    for comment in comments:
        new_comment = simple_preprocess(comment, deacc=True)
        final.append(new_comment)
    return final


def make_bigram_trigram(data_words):
    """
    Creates bigrams and trigrams from the data.

    Parameters:
        data_words (list): List of tokenized words.

    Returns:
        list: List of words with bigrams and trigrams.
    """
    bigrams_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=10)
    trigram_phrases = gensim.models.Phrases(bigrams_phrases[data_words], min_count=15, threshold=10)

    bigram = gensim.models.phrases.Phraser(bigrams_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = [bigram[doc] for doc in data_words]
    data_bigrams_trigrams = [trigram[doc] for doc in data_bigrams]
    return data_bigrams_trigrams


def compute_coherence_perplexity(corpus, texts, dictionary, params):
    """
        Computes coherence and perplexity of the LDA model.

        Parameters:
            corpus (list): List of document term matrices.
            texts (list): List of tokenized texts.
            dictionary (gensim.corpora.Dictionary): Gensim dictionary.
            params (dict): Parameters for LDA model.

        Returns:
            tuple: Coherence score, perplexity score, and LDA model.
    """
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
    """
        Performs random search for hyperparameter optimization.

        Parameters:
            param_grid (dict): Dictionary of parameter grid.
            n_iter (int): Number of iterations.

        Returns:
            list: List of random parameter combinations.
    """

    param_combinations = list(itertools.product(*param_grid.values()))
    return random.sample(param_combinations, n_iter)


def perform_random_search():
    """
    Performs random search for LDA model hyperparameters.
    """
    """
    # 1st Random Search - Online Learning
    # Best Params: {'num_topics': 10, 'update_every': 2, 'chunksize': 200, 'passes': 30, 'alpha': 'auto'} =	 Coherence: 0.5370863931633967
    param_grid = {'num_topics': [5, 7, 10],'update_every': [1, 2],'chunksize': [100, 200, 300], 'passes': [10, 20, 30],'alpha': ['symmetric', 'auto']}

    # 2nd Random Search - Batch Learning
    # Best Params: {'num_topics': 10, 'update_every': 1, 'chunksize': 300, 'passes': 20, 'alpha': 'auto'} =	 Coherence: 0.5683360578973087
    param_grid = {'num_topics': [5, 7, 10],'update_every': [0],'passes': [10, 20, 30],'alpha': ['symmetric', 'auto']}

    param_grid = {'num_topics': [5, 7, 10],'update_every': [1, 2],'chunksize': [200, 300],'passes': [20, 30],'alpha': ['auto'], 'eta': ["auto", "symmetric"]}

    param_grid = { 'num_topics': [5, 7, 10, 15], 'update_every': [1, 2],'chunksize': [100, 200, 300], 'passes': [10, 20, 30], 'alpha': ['symmetric', 'auto', 0.01, 0.1, 0.5], 'eta': ['symmetric', 'auto', 0.01, 0.1, 0.5]
    }
    """
    param_grid = {
        'num_topics': [5, 6, 7],
        'update_every': [1],
        'chunksize': [500, 600, 700],
        'passes': [40, 50],
        'alpha': [0.01, 0.05, 0.1],
        'eta': [0.01, 0.05, 0.1]
    }

    n_iter = 20  # Number of random parameter combinations to try
    param_combinations = random_search(param_grid, n_iter)
    param_names = list(param_grid.keys())
    results = []

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
            best_params = param_dict

    print(f'Best Params: {best_params} => Coherence: {best_coherence}')
    plot_search_results(results)


def plot_search_results(results):
    """
    Plots the results of the parameter search.

    Parameters:
        results (list): List of tuples containing (param_dict, coherence, perplexity).

    Returns:
        None
    """
    # Adjust the params and results to make the plot more readable
    params = [str(result[0]).replace('num_topics', 'n').replace('update_every', 'u').replace('chunksize', 'c').replace(
        'passes', 'p').replace('alpha', 'a').replace('eta', 'e') for result in results]
    coherence = [round(result[1], 3) for result in results]
    perplexity = [round(result[2], 3) for result in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1st y-axis for Coherence
    color = 'tab:blue'
    ax1.set_xlabel('Parameter Combinations')
    ax1.set_ylabel('Coherence', color=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightblue')
    ax1.scatter(params, coherence, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 2nd y-axis for Perplexity

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Perplexity', color=color)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightcoral')
    ax2.scatter(params, perplexity, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Adjust the ticks of y axises
    coherence_ticks = np.linspace(0.45, 0.52, num=10)
    ax1.set_yticks(coherence_ticks)

    perplexity_ticks = np.linspace(min(perplexity), max(perplexity), num=10)
    ax2.set_yticks(perplexity_ticks)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.set_xticks(range(len(params)))
    ax1.set_xticklabels(params, rotation=45, ha='right', fontsize=5)
    plt.title('Parameter Search Results for LDA')
    plt.subplots_adjust(left=0.2, bottom=0.3, top=0.9)  # Adjust the margins
    plt.savefig("../../outputs/nlp/topic_modeling/coherence_vs_perplexity.png")
    plt.show()


def run_lda_model(comments_df, corpus, id2word, data_bigrams_trigrams):
    """
    Runs the LDA model with specified hyperparameters.

    Parameters:
        comments_df (pd.DataFrame): DataFrame containing comments.
        corpus (list): List of document term matrices.
        id2word (gensim.corpora.Dictionary): Gensim dictionary.
        data_bigrams_trigrams (list): List of words with bigrams and trigrams.

    Returns:
        None
    """
    # Adjust hyperparameters with the best result from random search
    num_topics = 7
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=num_topics,
                         update_every=1,
                         random_state=0,
                         passes=40,
                         alpha=0.1,
                         eta=0.01,
                         chunksize=600)

    perplexity = lda_model.log_perplexity(corpus)
    print(f'Perplexity: {perplexity}')

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_bigrams_trigrams, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}')

    topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)
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
    topic_modeling()
