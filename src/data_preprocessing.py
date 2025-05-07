import pandas as pd

# Load the sentiment dataset
sentiment_data_path = 'data/training.1600000.processed.noemoticon.csv'
sentiment_df = pd.read_csv(sentiment_data_path, encoding='latin1', header=None)

# Check the first few rows to understand its structure
print("Sentiment Data Sample:")
print(sentiment_df.head())

# Check the number of columns
print("Number of columns:", sentiment_df.shape[1])

# Rename columns based on the number of columns in the dataset
if sentiment_df.shape[1] == 6:
    sentiment_df.columns = ['Sentiment', 'ID', 'Date', 'Query', 'User', 'Text']
else:
    print("Unexpected number of columns. Please check the dataset structure.")

# Clean the text (remove URLs, mentions, and unnecessary characters)
sentiment_df['Text'] = sentiment_df['Text'].str.replace(r'http\S+|www\S+', '', regex=True)  # Remove URLs
sentiment_df['Text'] = sentiment_df['Text'].str.replace(r'@\w+', '', regex=True)  # Remove mentions

# Map sentiment labels to binary values (0 = negative, 1 = positive)
sentiment_df['Sentiment'] = sentiment_df['Sentiment'].map({0: 0, 4: 1})

# Drop unnecessary columns
sentiment_df = sentiment_df[['Sentiment', 'Text']]

# Display the first few rows after cleaning
print("Cleaned Sentiment Data:")
print(sentiment_df.head())

# Load the twitch reviews dataset
twitch_data_path = 'data/twitch_reviews.csv'
twitch_df = pd.read_csv(twitch_data_path)

# Inspect the column names to understand the structure
print("Twitch Reviews Column Names:", twitch_df.columns)

# Use the correct column name 'content' for the review text
review_column = 'content'

# Clean the text (remove URLs and unnecessary characters)
twitch_df[review_column] = twitch_df[review_column].str.replace(r'http\S+|www\S+', '', regex=True)  # Remove URLs

# Handle missing values if necessary
twitch_df = twitch_df.dropna(subset=[review_column])  # Drop rows with missing reviews

# Display the cleaned data
print("Cleaned Twitch Reviews Data:")
print(twitch_df.head())

# Save the cleaned data to CSV files
sentiment_df.to_csv('data/preprocessed_sentiment.csv', index=False)
twitch_df.to_csv('data/preprocessed_twitch_reviews.csv', index=False)
