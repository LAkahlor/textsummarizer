import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define the CSV file path.
csv_file = "full_data.csv"

# Load the dataset.
data = pd.read_csv(csv_file)

# Preprocess the text data.
text_data = data["content"].tolist()  # Replace "content" with the column name containing the text data

# Handle NaN and empty values in the text data.
text_data = [str(text) if not pd.isna(text) else "" for text in text_data]

# Create the GPT-2 tokenizer and model.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensure the tokenizer and model use the same vocabulary
assert tokenizer.vocab_size == model.config.vocab_size

# Add a new padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Make sure the model is aware of the new token
model.resize_token_embeddings(len(tokenizer))

def pad_collate_fn(batch):
    # Function to pad the sequences in a batch to the same length
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return input_ids, attention_mask

# Tokenize and encode the text data.
print("Tokenizing and encoding text data...")
encoded_data = tokenizer(text_data, padding='longest', truncation=True, max_length=256, return_tensors="pt")

# Extract input_ids and attention_mask from encoded data.
input_ids = encoded_data["input_ids"]
attention_mask = encoded_data["attention_mask"]

# Create a TensorDataset.
dataset = TensorDataset(input_ids, attention_mask)

# Split the dataset into training and testing sets.
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

# Create a DataLoader for training and testing data.
batch_size = 8
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

# Define the loss function.
loss_function = torch.nn.CrossEntropyLoss()

# Train the model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print("Training the model...")
for epoch in range(10):
    print(f"Epoch {epoch + 1}:")

    for batch_idx, batch in enumerate(train_loader):
        batch = [t.to(device) for t in batch]
        input_ids, attention_mask = batch

        # Create labels
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100

        # Debug prints
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Max input id: {torch.max(input_ids)}")  # Add this line

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

print("Training completed!")

# Save the model.
torch.save(model.state_dict(), "my_model.pt")
print("Model saved.")
