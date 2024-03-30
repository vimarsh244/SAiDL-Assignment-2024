import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003")

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(dataset["train"].features["ner_tags"].feature.names))

# Define the MoE layer
class MoELayer(nn.Module):
    def __init__(self, input_size, num_experts, expert_size):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.expert_size = expert_size
        self.experts = nn.ModuleList([nn.Linear(input_size, expert_size) for _ in range(num_experts)])
        self.gating = nn.Linear(input_size, num_experts)

    def forward(self, x):
        gate_scores = self.gating(x)
        gate_weights = torch.softmax(gate_scores, dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(expert_outputs * gate_weights.unsqueeze(2), dim=1)

# Define the model
class NERModel(nn.Module):
    def __init__(self, base_model, num_experts, expert_size):
        super(NERModel, self).__init__()
        self.base_model = base_model
        self.moe_layer = MoELayer(base_model.config.hidden_size, num_experts, expert_size)
        self.classifier = nn.Linear(expert_size, base_model.config.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        moe_output = self.moe_layer(hidden_states)
        logits = self.classifier(moe_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, base_model.config.num_labels), labels.view(-1))
            return loss, logits
        return logits

# Prepare the data for training
data_collator = DataCollatorForTokenClassification(tokenizer)
train_dataset = dataset["train"].map(lambda example: tokenizer(example["tokens"], is_split_into_words=True, return_offsets_mapping=True, padding="max_length", truncation=True), batched=True)
eval_dataset = dataset["validation"].map(lambda example: tokenizer(example["tokens"], is_split_into_words=True, return_offsets_mapping=True, padding="max_length", truncation=True), batched=True)

# Define the training arguments and trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

model = NERModel(base_model, num_experts=4, expert_size=256)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()