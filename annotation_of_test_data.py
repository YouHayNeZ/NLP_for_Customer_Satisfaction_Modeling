from preprocessing import *

data = import_data('data/ryanair_reviews.csv')
data = create_datetime(data)
data = encode_categoricals(data)
data = drop_missing_target(data)
# Splitting the dataset into the Training set and Test set
X = data[['Comment title', 'Comment']]
y = data['Overall Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

# combine X_test  and y_test again
data_test = pd.concat([X_test, y_test], axis=1)

# save to csv
data_test.to_csv('data/test_data_to_be_labeled.csv', index=False)

# To Do: Create list with the labels for the test data (write: 1 = positive, -1 = negative, 0 = neutral)
# use 10 annotations per line to not lose track of where you are at
# labels for lines 2-118 (Johannes) [line 1 is header]
annotations_1_118 = [1, -1, -1, -1, -1, -1, 1, 1, -1, 1,
                     -1, 1, 1, 1, -1, -1, 1, -1, 1, 0,
                     -1, 0, -1, -1, 1, -1, -1, 0, -1, 0,
                     -1, 1, -1, -1, 1, -1, 1, 1, -1, -1,
                     -1, -1, -1, -1, 1, 1, -1, 1, -1, -1,
                     1, 1, -1, 1, 1, 1, -1, -1, -1, -1,
                     1, 1, -1, 0, -1, -1, -1, -1, -1, -1,
                     1, -1, 0, 1, 1, -1, 1, 1, 1, -1,
                     1, 1, -1, -1, -1, -1, 1, -1, -1, -1,
                     0, -1, -1, -1, 1, -1, -1, -1, 1, -1,
                     0, -1, -1, -1, -1, 1, 1, -1, -1, -1,
                     -1, 1, 1, -1, -1, 1, 1]
# My last review (included): Ryanair customer review,Return flight from Manchester to Rome £58

# labels for 119-238 (Niklas)
# to get started

annotations_119_238 = [1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1,
                       -1, -1, -1, -1, -1, 1, 1, -1, -1, 1,
                       -1, -1, -1, 0, -1, -1, 0, -1, 1, -1,
                       1, -1, -1, 0, 1, 1, 1, -1, 0, 0,
                       -1, -1, 1, 1, 1, -1, 0, -1, 1, 1,
                       -1, 1, -1, -1, -1, 0, -1, -1, -1, 1,
                       -1, 1, -1, 1, 1, -1, -1, 1, -1, 0,
                       -1, 1, 1, 1, 1, 1, -1, 1, 1, -1,
                       -1, 1, -1, 0, 0, -1, 1, 0, -1, -1,
                       -1, -1, 1, -1, -1, -1, -1, 1, -1, -1,
                       -1, -1, -1, 1, -1, -1, -1, 1, -1, -1,
                       1, -1, -1, 1, -1, -1, -1, -1]

# My last review (included) "customer service is shocking"

# labels for 239-355 (Pati)
annotations_239_355 = [0, 0, 0,
                       0]

# My frist review (included) "Ryanair Customer Review - The bad press surrounding Ryanair proved to be totally
# unjustified on my round trip Madrid-Cagliari-Madrid. Check- in and security was the quickest and most polite ever (
# especially at the dedicated check-in/security at Madrid. Boarding was orderly and uneventful (both in Madrid and
# Cagliari). We left on schedule and arrived early on both segments. The flight attendants were professional polite
# and friendly. No negative comments at all and I'll be only too happy to fly with Ryanair again."

# labels for 356-472 (Ece)
annotations_356_472 = [1, 1, 0, 1, -1, 1, 1, 1, -1, -1,
                       1, -1, -1, -1, -1, 1, -1, -1, 1, 1,
                       1, 1, 1, 1, -1, 1, 1, 1, 0, 1,
                       -1, 1, 1, -1, 1, 1, 1, -1, 1, -1,
                       -1, -1, -1, -1, 1, -1, -1, 0, 1, 1,
                       1, -1, 1, -1, 1, 1, 1, -1, 1, -1,
                       0, -1, -1, 0, -1, 1, 1, -1, 0, -1,
                       1, -1, -1, -1, -1, -1, 1, -1, -1, -1,
                       1, 0, 1, -1, 1, 1, -1, -1, -1, 1,
                       1, -1, 1, 1, -1, -1, -1, 1, -1, -1,
                       -1, -1, 1, 1, 1, 1, -1, -1, - 1, -1,
                       -1, -1, 1, -1, 1, 0, 0]

# My last review (included) "friendly cabin crew - Never been a huge fan of Ryanair from past experiences but I
# acknowledge they provide an A-B service at a generally good price and that is why they are so popular. I wasn't
# expecting much from this recent trip I made from Stansted to Gothenburg but I came away feeling I'd do it all again
# if this is what I can get for my money. On time flights, clean airplane, friendly cabin crew and excellent choice
# of on board products. Yes, they are a hard seller and the rules are what they are, but we should all know that by
# now, and for the price you pay nothing comes close. If you want a weekend away in Europe at a low price why would
# you look elsewhere?"

# concat annotations
# annotations = pd.concat([annotations_1_118, annotations_119_238, annotations_239_355, annotations_356_472])
