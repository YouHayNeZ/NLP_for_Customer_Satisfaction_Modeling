from openai import OpenAI
import matplotlib.pyplot as plt
import pandas as pd

### Important, this function has only been invoked once! We have used a personal openai key! So, the method does not work anymore ###
def classify_sentiments(api_key: str):
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load the data
    df = pd.read_csv("data/ryanair_reviews.csv")
    
    # Extract the list of comments
    list_of_comments = df["Comment"].to_list()
    
    # Initialize a list to hold the sentiment classifications
    openai_classification = []
    
    # Iterate over each comment and classify sentiment
    for comment in list_of_comments:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a researcher that specializes in sentiment analysis: I'm going to provide you with reviews and you have to classify them into positive (1), neutral (0), negative (-1). No need to explain yourself, I fully trust you. Your only output should be the correct number."},
                {"role": "user", "content": "I'm very satisfied with the service on board. We flew as a family. Both flights to Tenerife and Warszawa were on time. The food isn't cheap but worth it. Some people say that the seats are uncomfortable or there is not enough space, but found them perfectly adequate. The aircraft was in excellent condition. I will fly with Ryanair again."},
                {"role": "assistant", "content": "1"},
                {"role": "user", "content": f"{comment}"}
            ]
        )
        # Append the classification to the list
        openai_classification.append(response.choices[0].message.content)
    
    # Add the classifications to the DataFrame
    df["openai_sentiment"] = openai_classification
    
    # Save the DataFrame to a new CSV file
    df.to_csv("outputs/nlp/sentiment_analysis/openai_sentiment_analysis.csv")


def openai_plot_sentiment_proportions(df, sentiment_col):
    """
    Plots the sentiment proportions from the given DataFrame column using a pie chart.

    Parameters:
        df (pd.DataFrame): The DataFrame containing sentiment data.
        sentiment_col (str): The column name in the DataFrame which contains sentiment classifications.

    Returns:
        None
    """
    # Define the mapping from sentiment values to labels
    sentiment_labels = {1: 'positive', 0: 'neutral', -1: 'negative'}
    
    # Calculate sentiment proportions
    sentiment_proportions = df[sentiment_col].value_counts(normalize=True)

    # Map the sentiment values to their corresponding labels
    labels = [sentiment_labels[val] for val in sentiment_proportions.index]

    # Define lighter colors for the sentiment categories
    sentiment_colors = ['#ff6666', 'green', '#ffff99']

    # Plot the pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(
        sentiment_proportions,
        labels=labels,
        colors=sentiment_colors,
        explode=[0.1 if i == 0 else 0 for i in range(len(sentiment_proportions))],  # Explode the first slice
        autopct='%1.1f%%',
        startangle=180
    )
    plt.title("Sentiment Distribution Using OpenAI")

    # Show the plot
    plt.show()

# Call the function with the OpenAI API key
if __name__ == '__main__':
    classify_sentiments(api_key="")