
import pandas as pd
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


# Load the data
df = pd.read_csv("../../data/ryanair_reviews.csv")
sample_size = int(0.05 * len(df))
docs = df["Comment"].sample(n=sample_size).to_list()
single_string = " ".join(docs)
client = openai.OpenAI(api_key="")
"""
# Fine-tune topic representations with GPT
openai_embedder = OpenAIBackend(client, "text-embedding-ada-002")
prompt = f"You are a top topic modeling expert. I'm going to provide you customer reviews {docs} on RyanAir and you will have " \
         "to get the topics from them. One topic could for example be: 'Cancelled flights'. DO NOT FOCUS ON THE " \
         "destinations. Just respond with the topic"
"""
# representation_model = OpenAI(client, model="gpt-4o", chat=True)#, prompt=prompt)
# topic_model = BERTopic(representation_model=representation_model)#, embedding_model=openai_embedder)

# https://platform.openai.com/docs/guides/rate-limits/error-mitigation?context=tier-one
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

completion = completion_with_backoff(model="gpt-4o", messages=[
        {"role": "system",
         "content": "You are a researcher that specializes in topic modeling: I'm going to provide you with several review put into one string and you have to tell me what the main topics are. The topics should be seperated by commas. Possible topics could be: good customer service, bad customer service, issues with boarding, issues with bags. Make sure that the topics are distinctive"},
        {"role": "user",
         "content": "I'm very satisfied with the service on board. We flew as a family. Both flights to Tenerife and Warszawa were on time. The food isn't cheap but worth it. Some people say that the seats are uncomfortable or there is not enough space, but found them perfectly adequate. The aircraft was in excellent condition. I will fly with Ryanair again."},
        {"role": "assistant", "content": "Good service on board, On time, Good food, Adequate seats"},
        {"role": "user", "content": f"{single_string}"}
    ])

topics_first_run = completion.choices[0].message
print(topics_first_run)
with open('topics_first_run', 'w') as file:
    file.write(topics_first_run.content)

topics_second_run = completion_with_backoff(model="gpt-4o", messages=[
        {"role": "system",
         "content": f"You are a researcher that specializes in topic modeling: I'm going to provide you with several topics and I want to take a look at them critically. Make sure to respond only with a list of topics that are good. Seperate the items in teh list with a comma. Topics should be distinctive from each other. Each feature should be have a seperate topic (expect in cases where it makes sense as for example in 'Good food and drinks')."},
        {"role": "user", "content": f"Here are the topics: {topics_first_run.content}"}
    ])

print(topics_second_run.choices[0].message)
with open('topics_second_run', 'w') as file:
    file.write(topics_second_run.choices[0].message.content)

topics_per_comment = []

comments = df["Comment"].to_list()
for comment in comments:
    response_comment = completion_with_backoff(model="gpt-4o", messages=[
            {"role": "system",
             "content": f"You are a researcher that specializes in topic modeling: I'm going to provide you with a review and a list of possible topics and you are going to choose which topics fit best, it could either be one or more than one, put a comma in between different topics. Here is the list of topics: {topics_second_run}"},
            {"role": "user", "content": f"{comment}"}
        ]
    )
    
    print(comment)
    print(response_comment.choices[0].message)
    # Append the classification to the list
    topics_per_comment.append(response_comment.choices[0].message.content)

print(topics_per_comment)

df["topics"] = topics_per_comment
# Save the DataFrame to a new CSV file
df.to_csv("outputs/nlp/topic_modeling/openai_topic_modeling.csv")



