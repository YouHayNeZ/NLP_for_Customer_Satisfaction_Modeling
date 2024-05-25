import pandas as pd
import nltk
import spacy
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

# Ensure the necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load the spacy model
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


# Function to read data
def read_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data


# Function to preprocess text
def preprocess_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Lowercase
    text = text.lower()
    # Tokenization and Lemmatization
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)


# Function to plot word clouds for topics
def plot_wordclouds(lda_model, feature_names):
    plt.figure(figsize=(15, 10))
    for topic_idx, topic in enumerate(lda_model.components_):
        wordcloud = WordCloud(background_color='white')
        wordcloud.generate_from_frequencies({feature_names[i]: topic[i] for i in topic.argsort()[:-11:-1]})
        plt.subplot(2, 3, topic_idx + 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Topic #{topic_idx}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/plots/lda_topics.png")
    plt.show()


# Main function for LDA topic modeling
def lda_topic_modeling(file_path, num_topics=5):
    # Step 1: Read the data
    data = read_data(file_path)

    # Step 2: Preprocess the comments
    data['processed_comment'] = data['Comment'].apply(preprocess_text)

    # Step 3: Vectorize the comments
    vectorizer = CountVectorizer(max_df=0.70, min_df=100, stop_words='english')
    # max_df: the words that appear in more than x % of the texts will not be considered as topics
    # min_df: for the topic to be considered in the modelling it has to be present in at least 100 inputs
    dtm = vectorizer.fit_transform(data['processed_comment'])

    # Step 4: Apply LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(dtm)

    # Step 5: Print the topics and plot word clouds
    feature_names = vectorizer.get_feature_names_out()
    plot_wordclouds(lda, feature_names)

    # Step 6: Extract topic assignments for each comment
    topic_assignments = lda.transform(dtm)
    for i in range(num_topics):
        data[f"Topic_{i+1}_probability"] = topic_assignments[:, i]

    # Step 7: Extract topic name with highest probability for each row
    topic_names = [f"Topic_{i + 1}" for i in range(num_topics)]
    max_topic_indices = topic_assignments.argmax(axis=1)
    max_topic_names = [topic_names[idx] for idx in max_topic_indices]

    # Step 8: Add a new column with the topic name with the highest probability
    data['Max_Probability_Topic'] = max_topic_names

    # Step 7: Save results to a CSV file
    data.to_csv("data/lda_topics.csv", index=False)


if __name__ == "__main__":
    lda_topic_modeling("data/cleaned_comments.csv", num_topics=5)
