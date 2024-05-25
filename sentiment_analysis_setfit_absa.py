# python -m pip install -U "setfit[absa]"
# python -m spacy download en_core_web_lg

# Aspects
import spacy
import pandas as pd


from sentiment_analysis_textblob import read_data

# Load the English NER model
nlp = spacy.load("en_core_web_sm")


def extract_aspects(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Initialize empty lists for aspects and their labels
    aspects = []
    labels = []

    # Iterate through the entities in the processed text
    for ent in doc.ents:
        # Append the aspect and its label to the respective lists
        aspects.append(ent.text)
        labels.append(ent.label_)

    return aspects, labels


if __name__ == '__main__':
    data, comments = read_data()

    comments = comments.head(10)

    # Apply the extract_aspects function to each row of the 'Comment' column
    comments[['Aspects', 'Labels']] = comments['Comment'].apply(lambda x: pd.Series(extract_aspects(x)))

    comments.to_csv("data/setfit_temp.csv")