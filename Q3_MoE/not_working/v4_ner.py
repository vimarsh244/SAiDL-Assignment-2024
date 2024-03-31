import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report
import numpy as np

# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Mapping from label to index
# Mapping from label to index
tag_to_index = {"O": 0, "B-MISC": 1, "I-MISC": 2, "B-PER": 3, "I-PER": 4, "B-ORG": 5, "I-ORG": 6, "B-LOC": 7, "I-LOC": 8}

index_to_tag = {idx: tag for tag, idx in tag_to_index.items()}

# Define model architecture
class LSTM_MoE_NER(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_experts, output_dim, dropout=0.1):
        super(LSTM_MoE_NER, self).__init__()
        self.num_experts = num_experts
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.moe = nn.ModuleList([nn.Linear(hidden_dim*2, hidden_dim*2) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_dim*2, num_experts)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout(lstm_out1)
        
        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = torch.tanh(self.moe[i](lstm_out1))
            expert_outputs.append(expert_output)

        gate_output = torch.sigmoid(self.gate(lstm_out1))
        weighted_expert_outputs = torch.stack(expert_outputs, dim=-1) * gate_output.unsqueeze(-1)
        mixed_output = torch.sum(weighted_expert_outputs, dim=-1)

        lstm_out2, _ = self.lstm2(mixed_output)
        lstm_out2 = self.dropout(lstm_out2)
        output = self.fc(lstm_out2)
        return output
    
def map_tag_to_index(tag):
    return tag_to_index.get(tag, 0)  # Map unknown tags to index 0 (corresponds to "O")


# Custom collate function for padding sequences
def collate_fn(batch):
    sentences = [item["tokens"] for item in batch]
    tags = [item["ner_tags"] for item in batch]

    tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=False) for sent in sentences]
    max_len = max(len(sent) for sent in tokenized_sentences)

    # Pad tokenized sentences and tags
    padded_sentences = pad_sequence([torch.tensor(sent) for sent in tokenized_sentences], batch_first=True, padding_value=0)
    padded_tags = pad_sequence([torch.tensor([map_tag_to_index(tag) for tag in tags[i]]) for i in range(len(tags))], batch_first=True, padding_value=0)

    attention_masks = torch.where(padded_sentences != 0, torch.tensor(1), torch.tensor(0))

    return padded_sentences, padded_tags, attention_masks


# Dataset class
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {"tokens": self.dataset[idx]["tokens"], "ner_tags": self.dataset[idx]["ner_tags"]}

# Hyperparameters
INPUT_DIM = tokenizer.vocab_size
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_EXPERTS = 4
OUTPUT_DIM = len(tag_to_index)
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 5

# Split dataset into train and validation
train_dataset = NERDataset(dataset["train"])
val_dataset = NERDataset(dataset["validation"])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Initialize model, optimizer, and loss function
model = LSTM_MoE_NER(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_EXPERTS, OUTPUT_DIM, DROPOUT)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        inputs, targets, masks = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, OUTPUT_DIM), targets.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets, masks = batch
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, OUTPUT_DIM), targets.view(-1))
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=-1)
            all_preds.extend(preds[masks.bool()].cpu().numpy())
            all_targets.extend(targets[masks.bool()].cpu().numpy())
    val_loss /= len(val_loader)

    # Calculate metrics
    val_report = classification_report(all_targets, all_preds, target_names=index_to_tag.values())

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print("Validation Report:\n", val_report)
