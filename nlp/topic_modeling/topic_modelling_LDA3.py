import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
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


def lda_model_1(data_words):
    """
    This is the first LDA Model

    Perplexity: -7.354517288398419
    Coherence Score: 0.4374928025539059

    Saves the LDA visualization under output
    """
    id2word = corpora.Dictionary(data_words)

    corpus = []
    for word in data_words:
        new_word = id2word.doc2bow(word)
        corpus.append(new_word)

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,
                                                random_state=0,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha="auto")

    # Calculate Perplexity
    perplexity = lda_model.log_perplexity(corpus)
    print(f'Perplexity: {perplexity}')

    # Calculate Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}')

    lda_vis_data = gensimvis.prepare(lda_model, corpus, id2word, mds="mmds", R=10)
    pyLDAvis.save_html(lda_vis_data, '../../outputs/nlp/topic_modeling/lda_visualization_iteration_1.html')


def lda_model_2(data_words):
    """
        This is the 2md LDA Model

        Perplexity: -7.439872907017229
        Coherence Score: 0.4366026805598106

        Saves the LDA visualization under output
        """
    # bigrams and trigrams

    bigrams_phrases = gensim.models.Phrases(data_words, min_count=20, threshold=20)
    trigram_phrases = gensim.models.Phrases(bigrams_phrases[data_words], min_count=20, threshold=20)

    bigram = gensim.models.phrases.Phraser(bigrams_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = make_bigrams(data_words, bigram)
    data_bigrams_trigrams = make_trigrams(data_bigrams, trigram)

    id2word = corpora.Dictionary(data_bigrams_trigrams)

    texts = data_bigrams_trigrams

    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,
                                                random_state=0,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha="auto")

    # Calculate Perplexity
    perplexity = lda_model.log_perplexity(corpus)
    print(f'Perplexity: {perplexity}')

    # Calculate Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_bigrams_trigrams, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}')

    lda_vis_data = gensimvis.prepare(lda_model, corpus, id2word, mds="mmds", R=10)
    pyLDAvis.save_html(lda_vis_data, '../../outputs/nlp/topic_modeling/lda_visualization_iteration_2.html')


def lda_model_2(data_words):
    """
        This is the 2nd LDA Model
        Introducing bigrams and trigrams into the model

        Perplexity: -7.584156611502125
        Coherence Score: 0.4716788582154613

        Saves the LDA visualization under output
        """
    # bigrams and trigrams

    bigrams_phrases = gensim.models.Phrases(data_words, min_count=10, threshold=10)
    trigram_phrases = gensim.models.Phrases(bigrams_phrases[data_words], min_count=10, threshold=10)

    bigram = gensim.models.phrases.Phraser(bigrams_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = make_bigrams(data_words, bigram)
    data_bigrams_trigrams = make_trigrams(data_bigrams, trigram)

    id2word = corpora.Dictionary(data_bigrams_trigrams)

    texts = data_bigrams_trigrams

    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,
                                                random_state=0,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha="auto")

    perplexity = lda_model.log_perplexity(corpus)
    print(f'Perplexity: {perplexity}')

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_bigrams_trigrams, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}')

    lda_vis_data = gensimvis.prepare(lda_model, corpus, id2word, mds="mmds", R=10)
    pyLDAvis.save_html(lda_vis_data, '../../outputs/nlp/topic_modeling/lda_visualization_iteration_2.html')


def lda_model_3(df, data_words):
    """
        This is the 3rd LDA Model
        Introducing bigrams and trigrams into the model
        Introducing TF IDF Removal

        Perplexity: -7.2097663120338655
        Coherence Score: 0.5243934846325902

        Saves the LDA visualization under output
        """
    bigrams_phrases = gensim.models.Phrases(data_words, min_count=30, threshold=20)
    trigram_phrases = gensim.models.Phrases(bigrams_phrases[data_words], min_count=30, threshold=20)

    bigram = gensim.models.phrases.Phraser(bigrams_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = make_bigrams(data_words, bigram)
    data_bigrams_trigrams = make_trigrams(data_bigrams, trigram)

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

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=7,
                                                random_state=0,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha="auto")

    perplexity = lda_model.log_perplexity(corpus)
    print(f'Perplexity: {perplexity}')

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_bigrams_trigrams, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}')

    lda_vis_data = gensimvis.prepare(lda_model, corpus, id2word, mds="mmds", R=5)
    pyLDAvis.save_html(lda_vis_data, '../../outputs/nlp/topic_modeling/lda_visualization_iteration_3.html')

    # Save topic probability distribution and max probability topic
    topic_assignments = [lda_model.get_document_topics(bow, minimum_probability=0) for bow in corpus]
    num_topics = lda_model.num_topics

    for doc_idx, topic_dist in enumerate(topic_assignments):
        topic_probabilities = np.zeros(num_topics)
        for topic_num, prob in topic_dist:
            topic_probabilities[topic_num] = prob

        for i in range(num_topics):
            df.loc[doc_idx, f"Topic_{i + 1}_probability"] = topic_probabilities[i]

        max_topic_idx = topic_probabilities.argmax()
        df.loc[doc_idx, 'Max_Probability_Topic'] = f"Topic_{max_topic_idx + 1}"

    return df


def compute_coherence_perplexity(corpus, texts, dictionary, params):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=params['num_topics'],
                                                random_state=0,
                                                update_every=params['update_every'],
                                                chunksize=params['chunksize'],
                                                passes=params['passes'],
                                                alpha=params['alpha'])
    perplexity = lda_model.log_perplexity(corpus)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model_lda.get_coherence()
    return coherence, perplexity, lda_model


def lda_model_4(data_words):
    """
    This is the 4th LDA Model
    Grid search for hyperparameters and plotting results

    param_grid = {
        'num_topics': [5, 7, 10],
        'update_every': [1, 2],
        'chunksize': [100, 200],
        'passes': [10, 20],
        'alpha': ['symmetric', 'auto']
    }

    """

    bigrams_phrases = gensim.models.Phrases(data_words, min_count=30, threshold=20)
    trigram_phrases = gensim.models.Phrases(bigrams_phrases[data_words], min_count=30, threshold=20)

    bigram = gensim.models.phrases.Phraser(bigrams_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = make_bigrams(data_words, bigram)
    data_bigrams_trigrams = make_trigrams(data_bigrams, trigram)

    id2word = corpora.Dictionary(data_bigrams_trigrams)
    corpus = [id2word.doc2bow(text) for text in data_bigrams_trigrams]

    param_grid = {
        'num_topics': [7, 10],
        'update_every': [1, 2],
        'chunksize': [100, 200, 300],
        'passes': [10, 20, 30],
        'alpha': ['auto']
    }

    # Perform grid search
    param_combinations = list(itertools.product(*param_grid.values()))
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

    # Plot results
    coherences = [result[1] for result in results]
    perplexities = [result[2] for result in results]
    labels = [str(result[0]) for result in results]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Parameter Combination')
    ax1.set_ylabel('Coherence Score', color=color)
    ax1.plot(labels, coherences, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=90)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Perplexity', color=color)
    ax2.plot(labels, perplexities, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Grid Search LDA Model')
    plt.savefig('Only_Gensim_Grid_Search.png')
    plt.show()


def lda_model_5(df,data_words):
    """
    Model with the results of the grid search

    """
    bigrams_phrases = gensim.models.Phrases(data_words, min_count=30, threshold=20)
    trigram_phrases = gensim.models.Phrases(bigrams_phrases[data_words], min_count=30, threshold=20)

    bigram = gensim.models.phrases.Phraser(bigrams_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = make_bigrams(data_words, bigram)
    data_bigrams_trigrams = make_trigrams(data_bigrams, trigram)

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

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,
                                                random_state=0,
                                                update_every=1,
                                                chunksize=300,
                                                passes=10,
                                                alpha="auto")

    perplexity = lda_model.log_perplexity(corpus)
    print(f'Perplexity: {perplexity}')

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_bigrams_trigrams, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}')

    lda_vis_data = gensimvis.prepare(lda_model, corpus, id2word, mds="mmds", R=5)
    pyLDAvis.save_html(lda_vis_data, '../../outputs/nlp/topic_modeling/lda_visualization_iteration_final.html')

    # Save topic probability distribution and max probability topic
    topic_assignments = [lda_model.get_document_topics(bow, minimum_probability=0) for bow in corpus]
    num_topics = lda_model.num_topics

    for doc_idx, topic_dist in enumerate(topic_assignments):
        topic_probabilities = np.zeros(num_topics)
        for topic_num, prob in topic_dist:
            topic_probabilities[topic_num] = prob

        for i in range(num_topics):
            df.loc[doc_idx, f"Topic_{i + 1}_probability"] = topic_probabilities[i]

        max_topic_idx = topic_probabilities.argmax()
        df.loc[doc_idx, 'Max_Probability_Topic'] = f"Topic_{max_topic_idx + 1}"

    return df


if __name__ == '__main__':
    stopwords_list = stopwords.words("english")
    stopwords_list.extend(['from', 'fly', 'flight', 'ryanair', 'travel'])
    comments = pd.read_csv("../../outputs/nlp/sentiment_analysis/cleaned_comments.csv")
    if 'Unnamed: 0' in comments.columns:
        comments.drop(columns=["Unnamed: 0"], inplace=True)
    cleaned_comments = comments['cleaned_Comment']

    lemmatizated_comments = lemmatization(cleaned_comments)

    data_words = gen_words(lemmatizated_comments)
    data_words = remove_stopwords(data_words, stopwords_list)

    result_df = lda_model_3(comments, data_words)

    result_df.to_csv("../../outputs/nlp/topic_modeling/lda_topics3.csv", index=False)
