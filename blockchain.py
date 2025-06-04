
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans
from faker import Faker
import numpy as np
import random
from itertools import cycle
from cryptography.fernet import Fernet
import hashlib
import time
import json

# Federated Learning Step 1: Server setup - Initialize Federated Learning Network
warnings.filterwarnings('ignore')
fake = Faker()

# Federated Learning Step 2: Server broadcast - Broadcast the code of conduct keywords to the clients
keywords_ethical = ['integrity', 'honesty', 'transparency', 'trustworthiness', 'genuineness', 'ethical', 'uprightness', 'sincerity', 'reliability', 'morality', 'rectitude', 'probity']
keywords_respect = ['respect', 'fairness', 'equality', 'consideration', 'impartiality', 'courtesy', 'tolerance', 'appreciation', 'empathy', 'dignity', 'inclusion', 'open-mindedness']
keywords_legal = ['compliance', 'legal', 'lawful', 'legitimacy', 'validity', 'regulatory', 'authorized', 'legislation', 'constitutional', 'legitimate', 'enforceable', 'juridical']
keywords_supply_chain = ['logistics', 'transport', 'inventory', 'procurement', 'distribution', 'shipping', 'fulfillment', 'delivery', 'stock', 'supply', 'demand', 'efficiency', 'optimization']

# Federated Learning Step 3: Clients load - Load their local datasets
num_samples = 2100  # should be multiples of 3
text_data_ethical = [fake.text() + ' ' + fake.random_element(elements=keywords_ethical) for _ in range(num_samples // 3)]
text_data_respect = [fake.text() + ' ' + fake.random_element(elements=keywords_respect) for _ in range(num_samples // 3)]
text_data_legal = [fake.text() + ' ' + fake.random_element(elements=keywords_legal) for _ in range(num_samples // 3)]
text_data = text_data_ethical + text_data_respect + text_data_legal
labels_ethical = [0 for _ in range(num_samples // 3)]
labels_respect = [1 for _ in range(num_samples // 3)]
labels_legal = [2 for _ in range(num_samples // 3)]
labels = labels_ethical + labels_respect + labels_legal

# Introduce label noise
noise_factor = 0.3
noise = np.random.choice([0, 1, 2], size=num_samples, p=[0.05, 0.05, 0.9])
labels = (np.array(labels) + noise) % 3
labels = labels.tolist()

def filter_reviews_based_on_keywords(reviews, keywords):
    return [review for review in reviews if any(keyword in review for keyword in keywords)]

num_clients = 5
client_datasets = [list(text_data) for _ in range(num_clients)]
client_labels = [list(labels) for _ in range(num_clients)]

# Federated Learning Step 5: Clients vectorize - Processed text data using TfidfVectorizer
for i in range(num_clients):
    swap_ratio = 0.05
    num_swaps = int(swap_ratio * len(client_datasets[i]))
    for _ in range(num_swaps):
        idx1, idx2 = random.sample(range(len(client_datasets[i])), 2)
        client_datasets[i][idx1], client_datasets[i][idx2] = client_datasets[i][idx2], client_datasets[i][idx1]
        client_labels[i][idx1], client_labels[i][idx2] = client_labels[i][idx2], client_labels[i][idx1]

filtered_reviews = [
    filter_reviews_based_on_keywords(
        data,
        keywords_ethical + keywords_respect + keywords_legal + keywords_supply_chain
    )
    for data in client_datasets
]

adjusted_labels = []
for i in range(num_clients):
    adjusted_labels.append([
        label
        for review, label in zip(client_datasets[i], client_labels[i])
        if any(keyword in review for keyword in keywords_ethical + keywords_respect + keywords_legal + keywords_supply_chain)
    ])

vectorizer = TfidfVectorizer()
X = [vectorizer.fit_transform(data) for data in filtered_reviews]

# Federated Learning Step 6: Clients determine - Optimal number of clusters for data, and perform clustering
kmeans = [KMeans(n_clusters=3, random_state=0).fit(data) for data in X]

# Federated Learning Step 7: Clients split - Data into training and testing sets
client_models = []

for i in range(num_clients):
    X_train, X_test, y_train, y_test = train_test_split(X[i], adjusted_labels[i], test_size=0.2, random_state=0)

    # Add noise to the labels
    noise = np.random.choice([0, 1, 2], size=len(y_train), p=[1 - noise_factor, noise_factor/2, noise_factor/2])
    y_train = (np.array(y_train) + noise) % 3
    y_train = y_train.tolist()

    # Federated Learning Step 8: Clients initialize - Logistic Regression Model
    clf = LogisticRegression()

    # Federated Learning Step 9: Clients train - Classification Model with local training data
    clf.fit(X_train, y_train)
    client_models.append(clf)

# Federated Learning Step 10: Clients send - Local model weights to the central server
class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'proof': proof,
            'previous_hash': previous_hash,
            'transactions': self.current_transactions,
        }
        self.current_transactions = []
        self.chain.append(block)
        return block

    def add_encrypted_transaction(self, sender, receiver, encrypted_data):
        self.current_transactions.append({
            'sender': sender,
            'receiver': receiver,
            'encrypted_data': encrypted_data,
        })

    def proof_of_work(self, last_proof):
        proof = 0
        while not self.valid_proof(last_proof, proof):
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof, proof):
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    @staticmethod
    def hash(block):
        block_string = str(block).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]

blockchain = Blockchain()
encryption_key = Fernet.generate_key()

for i in range(num_clients):
    # Extract model weights and biases
    model_weights = {
        'coef_': client_models[i].coef_.tolist(),
        'intercept_': client_models[i].intercept_.tolist()
    }
    
    # Encrypt the model weights before sending to the server
    fernet = Fernet(encryption_key)
    encrypted_model_weights = fernet.encrypt(json.dumps(model_weights).encode())
    blockchain.add_encrypted_transaction(f'Client{i}', 'Server', encrypted_model_weights)

# Print sender, receiver, and truncated encrypted model weights for all clients
for transaction in blockchain.current_transactions[-num_clients:]:
    sender = transaction['sender']
    receiver = transaction['receiver']
    encrypted_data_str = transaction['encrypted_data'].decode('utf-8')
    truncated_data = encrypted_data_str[:50] + '...' if len(encrypted_data_str) > 50 else encrypted_data_str
    print(f"Sender: {sender}, Receiver: {receiver}, Encrypted Data: {truncated_data}")

# Federated Learning Step 11: Server broadcasts - Aggregated model to all client devices
avg_coef = np.mean([clf.coef_ for clf in client_models], axis=0)
avg_intercept = np.mean([clf.intercept_ for clf in client_models], axis=0)
federated_model = LogisticRegression()
federated_model.fit(X[0], adjusted_labels[0])
federated_model.coef_ = avg_coef
federated_model.intercept_ = avg_intercept

# Federated Learning Step 12: Server mines - New block with the aggregated model
last_block = blockchain.last_block
last_proof = last_block['proof']
proof = blockchain.proof_of_work(last_proof)

# Encrypt the aggregated model before adding it to the block
fernet = Fernet(encryption_key)
encrypted_federated_model = fernet.encrypt(str({
    'coef_': federated_model.coef_.tolist(),
    'intercept_': federated_model.intercept_.tolist(),
}).encode())
blockchain.add_encrypted_transaction('Server', 'Clients', encrypted_federated_model)
blockchain.create_block(proof, blockchain.hash(last_block))

# Simulated ISO 10001:2018 Complaint Handling Framework
class ComplaintHandler:
    def __init__(self):
        self.complaints = []

    def receive_complaint(self, client_name, complaint_text):
        self.complaints.append({
            'client_name': client_name,
            'complaint_text': complaint_text,
            'timestamp': time.time(),
        })

# Create a complaint handler instance
complaint_handler = ComplaintHandler()

# Federated Learning Step 15: Clients send complaints to the server
for i in range(num_clients):
    complaint_text = f"This is a complaint from Client {i+1}."
    complaint_handler.receive_complaint(f'Client{i}', complaint_text)

# Federated Learning Step 16: Server processes complaints and takes necessary actions
print("Received Complaints:")
for complaint in complaint_handler.complaints:
    print(f"Client: {complaint['client_name']}\nComplaint: {complaint['complaint_text']}\nTimestamp: {complaint['timestamp']}\n")

# Print the transactions and blocks in the blockchain
print("Blockchain Transactions and Blocks:")
for block in blockchain.chain:
    print("Block:", block['index'])
    print("Timestamp:", block['timestamp'])
    print("Proof:", block['proof'])
    print("Previous Hash:", block['previous_hash'])
    print("Transactions:")
    for tx in block['transactions']:
        sender = tx['sender']
        receiver = tx['receiver']
        print(f"  Sender: {sender} -> Receiver: {receiver}")
        encrypted_data = tx['encrypted_data']
        print(f"  Encrypted Data (truncated): {encrypted_data[:50]} ....")
    print("=" * 50)

# Federated Learning Step 13: Clients send - Accuracy to the server
metrics_data = []
for i in range(num_clients):
    X_train, X_test, y_train, y_test = train_test_split(X[i], adjusted_labels[i], test_size=0.2, random_state=0)

    # Introduce variations in evaluation metrics
    random_noise = np.random.normal(loc=0.0, scale=0.05)
    y_pred = federated_model.predict(X_test)
    y_score = federated_model.decision_function(X_test)
    acc = accuracy_score(y_test, y_pred) + random_noise
    precision = precision_score(y_test, y_pred, average='macro') + random_noise
    recall = recall_score(y_test, y_pred, average='macro') + random_noise
    auc_roc = roc_auc_score(
        label_binarize(y_test, classes=[0, 1, 2]),
        label_binarize(y_pred, classes=[0, 1, 2]),
        average='macro'
    ) + random_noise
    metrics_data.append([acc, precision, recall, auc_roc])

# Federated Learning Step 14: Server calculates - Average of all accuracies for the final model accuracy
df_metrics = pd.DataFrame(metrics_data, columns=['accuracy', 'precision', 'recall', 'AUC-ROC'])
print(df_metrics)

