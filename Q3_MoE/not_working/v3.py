import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import time

# Mixture-of-Experts Layer
class MoELayer(nn.Module):
    def __init__(self, input_size, expert_size, num_experts):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(input_size, expert_size) for _ in range(num_experts)])
        self.gating_net = nn.Linear(input_size, num_experts)

    def forward(self, x):
        gate_scores = self.gating_net(x)
        gate_probs = torch.softmax(gate_scores, dim=-1)
        expert_outputs = [expert(x) for expert in self.experts]
        combined_output = torch.stack(expert_outputs, dim=-1)
        weighted_output = (combined_output * gate_probs.unsqueeze(-2)).sum(-1)
        return weighted_output

# Model Architecture
class NERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_experts, expert_size):
        super(NERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.moe = MoELayer(hidden_dim, expert_size, num_experts)
        self.lstm2 = nn.LSTM(expert_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.moe(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x

class QAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_experts, expert_size):
        super(QAModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.moe = MoELayer(hidden_dim, expert_size, num_experts)
        self.lstm2 = nn.LSTM(expert_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.moe(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x

# Training Function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Evaluation Function
def evaluate(model, dataloader, criterion, metric, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    labels = []
    with torch.no_grad():
        for inputs, target in dataloader:
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            predictions.extend(torch.argmax(outputs, dim=-1).cpu().numpy().tolist())
            labels.extend(target.cpu().numpy().tolist())
    eval_loss = running_loss / len(dataloader)
    eval_metric = metric.compute(predictions=predictions, references=labels)
    return eval_loss, eval_metric

# Main Function
def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    conll_dataset = load_dataset("conll2003")
    squad_dataset = load_dataset("squad")

    # Set up tokenizers
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Tokenize datasets
    def tokenize_conll(examples):
        return tokenizer(examples["tokens"], is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)

    def tokenize_squad(examples):
        return tokenizer(examples["question"], examples["context"], max_length=512, truncation="only_second", return_tensors="pt", padding="max_length")

    tokenized_conll = conll_dataset.map(tokenize_conll, batched=True)
    tokenized_squad = squad_dataset.map(tokenize_squad, batched=True)

    # Set up data loaders
    ner_dataloader = DataLoader(tokenized_conll["train"], batch_size=32, shuffle=True, drop_last=True)
    qa_dataloader = DataLoader(tokenized_squad["train"], batch_size=32, shuffle=True, drop_last=True)

    # Hyperparameters
    vocab_size = tokenizer.vocab_size
    embedding_dim = 768  # BERT embedding size
    hidden_dim = 256
    num_experts = 4
    expert_size = 64
    num_epochs = 10
    lr = 0.001

    # Initialize models
    ner_model = NERModel(vocab_size, embedding_dim, hidden_dim, num_experts, expert_size).to(device)
    qa_model = QAModel(vocab_size, embedding_dim, hidden_dim, num_experts, expert_size).to(device)

    # Set up loss functions and optimizers
    ner_criterion = nn.CrossEntropyLoss()
    qa_criterion = nn.CrossEntropyLoss()
    ner_optimizer = optim.Adam(ner_model.parameters(), lr=lr)
    qa_optimizer = optim.Adam(qa_model.parameters(), lr=lr)

    # Set up metrics
    ner_metric = load_metric("seqeval")
    qa_metric = load_metric("squad")

    # Training and Evaluation
    ner_losses = []
    qa_losses = []
    ner_moe_scores = []
    qa_moe_scores = []
    ner_baseline_scores = []
    qa_baseline_scores = []

    for epoch in range(num_epochs):
        start_time = time.time()
        ner_loss = train(ner_model, ner_dataloader, ner_optimizer, ner_criterion, device)
        qa_loss = train(qa_model, qa_dataloader, qa_optimizer, qa_criterion, device)
        ner_val_loss, ner_moe_score = evaluate(ner_model, ner_dataloader, ner_criterion, ner_metric, device)
        qa_val_loss, qa_moe_score = evaluate(qa_model, qa_dataloader, qa_criterion, qa_metric, device)
        end_time = time.time()

        print(f'Epoch {epoch+1}/{num_epochs}, NER Loss: {ner_loss:.4f}, QA Loss: {qa_loss:.4f}, NER Val Loss: {ner_val_loss:.4f}, QA Val Loss: {qa_val_loss:.4f}, Time: {end_time-start_time:.2f}s')

        ner_losses.append(ner_loss)
        qa_losses.append(qa_loss)
        ner_moe_scores.append(ner_moe_score)
        qa_moe_scores.append(qa_moe_score)

        # Evaluate baseline models
        ner_baseline_model = NERModel(vocab_size, embedding_dim, hidden_dim, 1, hidden_dim).to(device)
        qa_baseline_model = QAModel(vocab_size, embedding_dim, hidden_dim, 1, hidden_dim).to(device)
        ner_baseline_loss, ner_baseline_score = evaluate(ner_baseline_model, ner_dataloader, ner_criterion, ner_metric, device)
        qa_baseline_loss, qa_baseline_score = evaluate(qa_baseline_model, qa_dataloader, qa_criterion, qa_metric, device)
        ner_baseline_scores.append(ner_baseline_score)
        qa_baseline_scores.append(qa_baseline_score)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(ner_losses, label='MoE')
    plt.plot([ner_baseline_scores[0]['overall_f1'] for _ in range(num_epochs)], label='Baseline')
    plt.title('NER Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot([score['overall_f1'] for score in ner_moe_scores], label='MoE')
    plt.plot([score['overall_f1'] for score in ner_baseline_scores], label='Baseline')
    plt.title('NER F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(qa_losses, label='MoE')
    plt.plot([qa_baseline_scores[0]['f1'] for _ in range(num_epochs)], label='Baseline')
    plt.title('QA Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot([score['f1'] for score in qa_moe_scores], label='MoE')
    plt.plot([score['f1'] for score in qa_baseline_scores], label='Baseline')
    plt.title('QA F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()