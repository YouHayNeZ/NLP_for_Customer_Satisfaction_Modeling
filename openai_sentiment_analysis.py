from openai import OpenAI
import pandas as pd

client = OpenAI(
  api_key="", #add key here
)
df = pd.read_csv("data/ryanair_reviews.csv")

list_of_comments = df["Comment"].to_list()
openai_classification = []
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
    openai_classification.append(response.choices[0].message.content)
print(openai_classification)
df["openai_sentiment"] = openai_classification
df.to_csv("data/openai_sentiment_analysis")