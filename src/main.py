import os
import torch
import pandas as pd
from dnn_model import SimpleDNN
from bert_sentiment_analysis import train_sentiment_model  # or your BERT inference function

def main():
    # Dynamically build paths based on project structure
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Corrected _file_ to __file__
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')

    # Load your cleaned Twitch reviews (for BERT sentiment analysis)
    twitch_reviews_path = os.path.join(data_dir, 'twitch_reviews.csv')
    twitch_reviews = pd.read_csv(twitch_reviews_path)
    print(f"Loaded Twitch reviews: {twitch_reviews.shape}")

    # Load the Sentiment140 dataset (for QoS or additional sentiment tasks)
    sentiment140_path = os.path.join(data_dir, 'training.1600000.processed.noemoticon.csv')  # Updated path
    sentiment140 = pd.read_csv(sentiment140_path, encoding='latin-1', header=None)
    print(f"Loaded Sentiment140 dataset: {sentiment140.shape}")

    # Example: Prepare features for DNN (replace with your actual logic)
    # Here, we use the length of the review as a placeholder feature
    twitch_reviews['feature'] = twitch_reviews['content'].apply(lambda x: len(str(x).split()))
    X = torch.tensor(twitch_reviews['feature'].values).float().view(-1, 1)
    y = torch.tensor([1 if len(str(x)) % 2 == 0 else 0 for x in twitch_reviews['content']]).long()

    # Split into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train DNN
    model = SimpleDNN(input_size=1, hidden_size=64, output_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Evaluate DNN
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Run BERT sentiment analysis on cleaned reviews
    # (Assume you have a function for this in bert_sentiment_analysis.py)
    print("Running BERT sentiment analysis on Twitch reviews...")
    # Example: sentiment_scores = run_bert_sentiment(twitch_reviews['cleaned_content'])
    # print(sentiment_scores.head())

    # You can now combine DNN predictions and BERT sentiment for your QoS logic

if __name__ == "__main__":  # Corrected _name_ to __name__
    main()
