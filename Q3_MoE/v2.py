import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForQuestionAnswering
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_experts = 4
expert_hidden_size = 256
gating_hidden_size = 128
num_layers = 2
dropout_rate = 0.2
learning_rate = 1e-3
batch_size = 32
num_epochs = 10

# MoE Layer
class MoELayer(nn.Module):
    def __init__(self, input_size, expert_size, num_experts, gating_size):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(input_size, expert_size) for _ in range(num_experts)])
        self.gating = nn.Linear(input_size, num_experts)

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_outputs = self.gating(x).softmax(dim=1)
        output = torch.bmm(gate_outputs.unsqueeze(1), expert_outputs).squeeze(1)
        return output

# Model
class NLPModel(nn.Module):
    def __init__(self, input_size, expert_size, num_experts, gating_size, num_layers, dropout_rate, output_size):
        super(NLPModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, expert_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.moe = MoELayer(expert_size, expert_size, num_experts, gating_size)
        self.lstm2 = nn.LSTM(expert_size, expert_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.output_layer = nn.Linear(expert_size, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.moe(x)
        x, _ = self.lstm2(x)
        x = self.output_layer(x)
        return x

# Load datasets
def load_conll2003_dataset():
    dataset = load_dataset("conll2003")
    
    # Preprocess the dataset
    def preprocess_function(examples):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Tokenize the text
        tokenized_inputs = tokenizer(examples["tokens"], padding="max_length", truncation=True, is_split_into_words=True)
        
        # Create the labels
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    dataset = dataset.map(preprocess_function, batched=True)
    return dataset

# Custom collate function for NER task
def collate_fn_ner(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}



def load_squad_dataset():
    dataset = load_dataset("squad")
    return dataset

# Custom collate function for NER task
def collate_fn_ner(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([torch.tensor(item['ner_tags']) for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Custom collate function for QA task
def collate_fn_qa(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    start_positions = torch.tensor([item['start_position'] for item in batch])
    end_positions = torch.tensor([item['end_position'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'start_positions': start_positions, 'end_positions': end_positions}

# Train and evaluate models
def train_and_evaluate(model, dataset, task, num_epochs, batch_size, learning_rate):
    if task == "ner":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model_for_task = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(dataset["train"].features["ner_tags"].feature.names))
        output_size = len(dataset["train"].features["ner_tags"].feature.names)
        collate_fn = collate_fn_ner
    elif task == "qa":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model_for_task = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        output_size = 2
        collate_fn = collate_fn_qa

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, collate_fn=collate_fn)

    train_losses, val_losses, test_losses = [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if task == "ner":
                labels = batch["labels"].to(device)
                output = model(input_ids, attention_mask=attention_mask)
                loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
            elif task == "qa":
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                output = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                loss = (output.start_logits, output.end_logits)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                if task == "ner":
                    labels = batch["labels"].to(device)
                    output = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
                elif task == "qa":
                    start_positions = batch["start_positions"].to(device)
                    end_positions = batch["end_positions"].to(device)
                    output = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                    loss = (output.start_logits, output.end_logits)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                if task == "ner":
                    labels = batch["labels"].to(device)
                    output = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
                elif task == "qa":
                    start_positions = batch["start_positions"].to(device)
                    end_positions = batch["end_positions"].to(device)
                    output = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                    loss = (output.start_logits, output.end_logits)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")

    return train_losses, val_losses, test_losses

# Evaluate and compare models
def evaluate_and_compare(baseline_model, moe_model, dataset, task):
    print(f"Evaluating Baseline Model on {task.upper()} task:")
    baseline_train_losses, baseline_val_losses, baseline_test_losses = train_and_evaluate(baseline_model, dataset, task, num_epochs, batch_size, learning_rate)

    print(f"Evaluating MoE-augmented Model on {task.upper()} task:")
    moe_train_losses, moe_val_losses, moe_test_losses = train_and_evaluate(moe_model, dataset, task, num_epochs, batch_size, learning_rate)

    # Plot the results
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, baseline_train_losses, label='Baseline Train Loss')
    plt.plot(epochs, baseline_val_losses, label='Baseline Val Loss')
    plt.plot(epochs, baseline_test_losses, label='Baseline Test Loss')
    plt.plot(epochs, moe_train_losses, label='MoE Train Loss')
    plt.plot(epochs, moe_val_losses, label='MoE Val Loss')
    plt.plot(epochs, moe_test_losses, label='MoE Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{task.upper()} Task - Baseline vs MoE-augmented Model')
    plt.legend()
    plt.show()

# Example usage
# Named Entity Recognition (NER) on CoNLL-2003 dataset
conll2003_dataset = load_conll2003_dataset()
ner_output_size = len(conll2003_dataset["train"].features["labels"].feature.names)
baseline_ner_model = NLPModel(input_size=768, expert_size=expert_hidden_size, num_experts=1, gating_size=gating_hidden_size, num_layers=num_layers, dropout_rate=dropout_rate, output_size=ner_output_size)
moe_ner_model = NLPModel(input_size=768, expert_size=expert_hidden_size, num_experts=num_experts, gating_size=gating_hidden_size, num_layers=num_layers, dropout_rate=dropout_rate, output_size=ner_output_size)
evaluate_and_compare(baseline_ner_model, moe_ner_model, conll2003_dataset, "ner")

# Question Answering on SQuAD 1.1 dataset
squad_dataset = load_squad_dataset()
qa_output_size = 2
baseline_qa_model = NLPModel(input_size=768, expert_size=expert_hidden_size, num_experts=1, gating_size=gating_hidden_size, num_layers=num_layers, dropout_rate=dropout_rate, output_size=qa_output_size)
moe_qa_model = NLPModel(input_size=768, expert_size=expert_hidden_size, num_experts=num_experts, gating_size=gating_hidden_size, num_layers=num_layers, dropout_rate=dropout_rate, output_size=qa_output_size)
evaluate_and_compare(baseline_qa_model, moe_qa_model, squad_dataset, "qa")