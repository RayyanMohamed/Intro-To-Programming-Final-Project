# RAYYAN MOHAMED FINAL PROJECT
# THIS CODE SHOULD BE RUN ON A COLAB NOTEBOOK

# Install and import packages
! pip install datasets
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
dataset = load_dataset("climate_fever")

len(dataset)
dataset["test"][0]
len(dataset["test"])
print(dataset)
print(dataset["test"]["claim"])

for d in dataset["test"]["evidences"][0]:
  print(d["evidence_label"])
  print(d["votes"])
  
# Create variables and copy contents for claims, claim_label, and claim_id
claims=dataset["test"]["claim"]
label=dataset["test"]["claim_label"]
claim_id=dataset["test"]["claim_id"]

# Create dataframe with claims and claim_labels
d={"claims":claims,"labels":label}
df=pd.DataFrame(data=d)

claim_id_list, claim_list, claim_label_list, evidence_label_list, votes_list = [], [], [], [], []


for i in range(len(df)):
  claim_id=dataset["test"]["claim_id"][i]
  claim=dataset["test"]["claim"][i]
  claim_label=dataset["test"]["claim_label"][i]
  for d in dataset["test"]["evidences"][i]:
    claim_id_list.append(claim_id)
    claim_list.append(claim)
    claim_label_list.append(claim_label)
    evidence_label_list.append(d["evidence_label"])
    votes_list.append(d["votes"])

a = 0
b = 0
c = 0
d = 0

a1 = 0
b1 = 0
c1 = 0
d1 = 0

labels = []
length = []

supports = []
refutes = []
not_enough_info = []
disputed = []

for i in range(len(df)):
  if(label[i] == 0):
    # Supports
    a = a + len(dataset["test"]["claim"][i])
    a1 = a1 + 1
    length.append(len(dataset["test"]["claim"][i]))
    labels.append(0)
    supports.append(len(dataset["test"]["claim"][i]))
  if(label[i] == 1):
    # Refutes
    b = b + len(dataset["test"]["claim"][i])
    b1 = b1 + 1
    length.append(len(dataset["test"]["claim"][i]))
    labels.append(1)
    refutes.append(len(dataset["test"]["claim"][i]))
  if(label[i] == 2):
    # Not enough information
    c = c + len(dataset["test"]["claim"][i])
    c1 = c1 + 1
    length.append(len(dataset["test"]["claim"][i]))
    labels.append(2)
    not_enough_info.append(len(dataset["test"]["claim"][i]))
  if(label[i] == 3):
    # Disputed
    d = d + len(dataset["test"]["claim"][i])
    d1 = d1 + 1
    length.append(len(dataset["test"]["claim"][i]))
    labels.append(3)
    disputed.append(len(dataset["test"]["claim"][i]))

print(a, ": ", a1)
print(b, ": ", b1)
print(c, ": ", c1)
print(d, ": ", d1)

print("Supports: ", 83754/654)
print("Refutes: ", 27698/253)
print("Not Enough Information: ", 59060/474)
print("Disputed: ", 18909/154)

plt.hist(supports, alpha = 0.5)
plt.hist(not_enough_info, alpha = 0.5)
plt.hist(disputed, alpha = 0.5)
plt.hist(refutes, alpha = 0.5)

plt.title('Frequency of Claim Length')
plt.xlabel('Claim Length (# of characters)')
plt.ylabel('Number of Occurences')

legend = ['Supports', 'Refutes', 'NEI', 'Disputes']
plt.legend(legend)

plt.style.use('seaborn')
plt.title('Claim Length vs. Claim Label')
plt.xlabel('Claim Length (# of characters)')
plt.ylabel('Claim Label (0, 1, 2, 3)')
plt.scatter(length, labels, edgecolor = 'black', linewidth = 1, alpha = 0.2)

# Create a data frame with claim_id, claim, claim_label, evidence_label, and votes
data={'claim_id' : claim_id_list, 'claim' : claim_list, 'claim_label' : claim_label_list, 'evidence_label' : evidence_label_list, 'votes' : votes_list}
df_evidence = pd.DataFrame(data=data)

# Print first five rows of the data frame
df_evidence.head()

print(votes_list)

df.labels.value_counts()

# Create a table with claim_id and claim_label
data_labels = {
    'Claim ID' : claim_id,
    'Claim Label' : label
}
data_table = pd.DataFrame(data_labels)
data_table

# Create new variable with length of claim text for each claim and plot
claim_lens=[len(text) for text in df.claims.values]
plt.hist(claim_lens)

# Model: claim / claim label

# Print length of dataset
len(dataset["test"])

# Split dataset intp train and test
from datasets import Dataset, DatasetDict
dataset = Dataset.from_pandas(df).train_test_split(test_size = 0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Print length of train dataset
len(train_dataset)

# Print first element in train dataset 
train_dataset[0]

# Print length of test dataset
len(test_dataset)

# Print first element in test dataset
test_dataset[0]

# Load tokenizer
! pip install transformers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Define preprocess function to run tokenizer on claims
def preprocess_function(examples):
    return tokenizer(examples["claims"], truncation=True)

# Tokenize train and test datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Use data collator for creating training batches
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load DistilBERT model
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

# Import module to measure classification performance
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def compute_metrics(pred):
    """
    this function is used within the model to calculate the accuracy, f1, precision, recall
    Args:
        param1 (int): either 1 or 0
    Returns:
        float: decimal values for accuracy, f1, precision, recall.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # f1: average of precision and recall
    f1 = f1_score(labels, preds, average="weighted")
    # accuracy: number of correctly predicted data points out of all data points
    acc = accuracy_score(labels, preds)
    # precision: number of true positives / number of false positives
    precision = precision_score(labels, preds, average="weighted")
    # recall: ratio of correctly classified positive samples to all positive samples
    recall = recall_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
  
! pip install huggingface_hub

from huggingface_hub import notebook_login

notebook_login()

training_args = TrainingArguments(
    output_dir="./model",
    push_to_hub = True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset, 
    compute_metrics = compute_metrics,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Run prediction on model using tokenized test dataset
predictions = trainer.predict(tokenized_test_dataset)

# Print model prediction metrics
trainer.evaluate()

# Get accuracy of model
# eval_loss: testing loss
# eval_accuracy: accuracy of testing
# eval_f1: combination of precision and recall
# eval_precision: 
# eval_recall: 

import numpy as np
from datasets import load_metric
metric = load_metric('accuracy')
preds = np.argmax(predictions.predictions, axis=1)
metric.compute(predictions=preds, references=predictions.label_ids)

# Save model checkpoints, weights, and other parameters to hugging face hub

trainer.push_to_hub()

# Install and import packages
! pip install datasets
import pandas as pd
import matplotlib.pyplot as plt

# Load climate_fever data 
from datasets import load_dataset
dataset = load_dataset("climate_fever")

claims=dataset["test"]["claim"]
label=dataset["test"]["claim_label"]
claim_id=dataset["test"]["claim_id"]

# Create dataframe with claims and claim_labels
d={"claims":claims,"labels":label}
df=pd.DataFrame(data=d)

claim_id_list, claim_list, claim_label_list, evidence_label_list, votes_list, evidence_list = [], [], [], [], [], []

for i in range(len(df)):
  claim_id=dataset["test"]["claim_id"][i]
  claim=dataset["test"]["claim"][i]
  claim_label=dataset["test"]["claim_label"][i]
  for d in dataset["test"]["evidences"][i]:
    claim_id_list.append(claim_id)
    claim_list.append(claim)
    claim_label_list.append(claim_label)
    evidence_label_list.append(d["evidence_label"])
    votes_list.append(d["votes"])
    evidence_list.append(d["evidence"])
    
# Create a data frame with claim_id, claim, claim_label, evidence_label, and votes
data={'claim_id' : claim_id_list, 'claim' : claim_list, 'claim_label' : claim_label_list, 'evidence' : evidence_list, 'evidence_label' : evidence_label_list, 'votes' : votes_list}
df_evidence = pd.DataFrame(data=data)

df_evidence.head()

df_evidence["claim_evidence"] = df_evidence["claim"] + "||" + df_evidence["evidence"]
del df_evidence["claim_id"]
del df_evidence["claim"]
del df_evidence["claim_label"]
del df_evidence["evidence"]
del df_evidence["votes"]
df_evidence["labels"] = df_evidence["evidence_label"]
del df_evidence["evidence_label"]
df_evidence.head(20)

# Split dataset intp train and test
from datasets import Dataset, DatasetDict
dataset = Dataset.from_pandas(df_evidence).train_test_split(test_size = 0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Print length of train dataset
len(train_dataset)

# Print first element in train dataset 
train_dataset[0]

# Print length of test dataset
len(test_dataset)

# Print first element in test dataset
test_dataset[0]

# Define preprocess function to run tokenizer on claims
def preprocess_function(examples):
    #return tokenizer(examples["claim"], examples["evidence"], examples["claim_evidence"], truncation=True)
    return tokenizer(examples["claim_evidence"], truncation=True)
  
# Tokenize train and test datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Use data collator for creating training batches
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load DistilBERT model
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

! pip install huggingface_hub

from huggingface_hub import notebook_login

notebook_login()

# Train model

training_args = TrainingArguments(
    output_dir="./model",
    push_to_hub = True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset, 
    compute_metrics = compute_metrics,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Run prediction on model using tokenized test dataset
predictions = trainer.predict(tokenized_test_dataset)

# Print model prediction metrics
trainer.evaluate()

# Get accuracy of model
# eval_loss: testing loss
# eval_accuracy: accuracy of testing
# eval_f1: combination of precision and recall
# eval_precision: 
# eval_recall: 

import numpy as np
from datasets import load_metric
metric = load_metric('accuracy')
preds = np.argmax(predictions.predictions, axis=1)
metric.compute(predictions=preds, references=predictions.label_ids)

# Save model checkpoints, weights, and other parameters to hugging face hub

trainer.push_to_hub()

# Download tokenizer and pretrianed model from my hugging face hub
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("IntroToProgramming/model")
model = AutoModelForSequenceClassification.from_pretrained("IntroToProgramming/model")

# Create a classifier using downloaded model and tokenizer and test model with new claims
from transformers import pipeline
classifier = pipeline("text-classification", model = model, tokenizer = tokenizer)
classifier("Global warming is driving polar bears toward extinction||Rising global temperatures, caused by the greenhouse effect, contribute to habitat destruction, endangering various species, such as the polar bear.")
#classifier("Global warming is causing more extreme weather events like floods and droughts")
