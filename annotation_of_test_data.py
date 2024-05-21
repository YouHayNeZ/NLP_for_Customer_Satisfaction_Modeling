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

# labels for lines 2-118 (Johannes) [line 1 is header]
annotations_1_118 = [1, -1, -1, -1, -1, -1, 1, 1, -1, 1,
                     -1, 1, 1, 1, -1, -1, 1, -1, 1, 0,
                     -1, 0, -1, -1, 1, -1, -1, 0, -1, 0,
                     -1, 1, -1, -1, 1, -1, 1, 1, -1, -1,
                     -1, -1, -1, -1 1, 1, -1, 1, -1, -1,
                     1, 1, -1, 1, 1, 1, -1, -1, -1, -1,
                     1, 1, -1, 0, -1, -1, -1, -1, -1, -1,
                     1, -1, 0, 1, 1, -1, 1, 1, 1, -1,
                     1, 1, -1, -1, -1, -1, 1, -1, -1, -1,
                     0, -1, -1, -1, 1, -1, -1, -1, 1, -1,
                     0, -1, -1, -1, -1, 1, 1, -1, -1, -1,
                     -1, 1, 1, -1, -1, 1, 1]

# use 10 annotations per line to not lose track of where you are at

# labels for 119-236 (Niklas)
annotations_119_236 = [0, 0, 0,
                       0]

# labels for 237-355 (Pati)
annotations_237_355 = [0, 0, 0,
                       0]

# labels for 356-472 (Ece)
annotations_356_472 = [0, 0, 0,
                       0]

# concat annotations
annotations = pd.concat([annotations_1_118, annotations_119_236, annotations_237_355, annotations_356_472])
