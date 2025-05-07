import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Function to encode reviews
def encode_reviews(texts, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    return {
        'input_ids': torch.cat(input_ids, dim=0),
        'attention_mask': torch.cat(attention_masks, dim=0)
    }

# Train the BERT model
def train_sentiment_model(batch_size=16, epochs=2):
    # Load processed data
    df = pd.read_csv('../data/processed_twitch_reviews.csv')
    
    # Create sentiment labels (similar to your DNN approach)
    df['sentiment'] = (df['reviewId'].apply(lambda x: 1 if isinstance(x, str) and len(x) % 2 == 0 else 0))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_content'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_encodings = encode_reviews(X_train)
    test_encodings = encode_reviews(X_test)
    
    train_dataset = TensorDataset(
        train_encodings['input_ids'], 
        train_encodings['attention_mask'],
        torch.tensor(y_train.values)
    )
    
    test_dataset = TensorDataset(
        test_encodings['input_ids'], 
        test_encodings['attention_mask'],
        torch.tensor(y_test.values)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} completed")
    
    return model