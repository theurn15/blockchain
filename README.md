# blockchain

README

A Novel Framework for Integrating Blockchain-Driven Federated Learning with Neural Networks in E‑Commerce.

Overview:

This repository contains a limited demo implementation based on the framework presented in:

Alshareet, O., Awasthi, A. “A Novel Framework for Integrating Blockchain-Driven Federated Learning with Neural Networks in E‑Commerce.” Journal of Network and Systems Management 33, 56 (2025). https://doi.org/10.1007/s10922-025-09928-x


**Anyone using this code should reference the above paper.**

The code simulates federated learning steps combined with blockchain transactions and a complaint handling framework, using logistic regression as the classification model.

Dependencies:
- Python 3.x
- pandas
- seaborn
- matplotlib
- scikit-learn
- faker
- cryptography
- numpy

Install dependencies via:
```
pip install pandas seaborn matplotlib scikit-learn faker cryptography numpy
```

Code Structure:
1. **Data Generation and Preprocessing**  
   - Simulated text data generated with Faker, embedding keywords for ethical, respect, legal, and supply chain categories.
   - Introduces label noise and filters reviews based on keywords.

2. **Federated Learning Steps**  
   - **Step 1: Server setup**  
     Initialize the federated learning network and broadcast code of conduct keywords.  
   - **Step 2: Client data load**  
     Each client generates a local dataset of synthetic reviews.  
   - **Step 3: Data processing**  
     Clean, tokenize, and add noise to labels to simulate realistic scenarios.  
   - **Step 4: Vectorization**  
     Clients use `TfidfVectorizer` to convert text into feature vectors.  
   - **Step 5: Clustering**  
     Clients determine optimal clusters using `KMeans` (for demonstration).  
   - **Step 6: Train/Test Split**  
     Each client splits their data into training and testing sets, adding further label noise.  
   - **Step 7: Model Initialization**  
     Clients initialize a `LogisticRegression` model (logistic regression implementation of the framework).  
   - **Step 8: Local Training**  
     Clients train their local logistic regression models.  
   - **Step 9: Encryption & Blockchain**  
     Each client encrypts local model weights using `Fernet` and sends them as transactions to a simulated `Blockchain` class.  
   - **Step 10: Aggregation**  
     The server aggregates model weights (averaging coefficients and intercepts) into a federated logistic regression model.  
   - **Step 11: Mining**  
     The server performs a proof-of-work to add a new block containing the encrypted federated model.  
   - **Step 12: Evaluation**  
     Clients evaluate the aggregated model locally, computing accuracy, precision, recall, and AUC-ROC with random variation.  
     - Saves confusion matrix and ROC curve plots as PDF files for each client.  
   - **Step 13: Complaint Handling**  
     Implements a simulated ISO 10001:2018 complaint handling framework via a `ComplaintHandler` class.  
   - **Step 14: Metrics Reporting**  
     The server calculates average evaluation metrics across all clients and prints a DataFrame.

3. **Blockchain Class**  
   - A minimal blockchain implementation with methods for:
     - Creating blocks (`create_block`)
     - Adding encrypted transactions (`add_encrypted_transaction`)
     - Proof-of-work (`proof_of_work` & `valid_proof`)
     - Hashing blocks (`hash`)
     - Accessing the last block (`last_block` property)

4. **ComplaintHandler Class**  
   - Simulated complaint handling:  
     - `receive_complaint(client_name, complaint_text)` stores incoming complaints for later processing.  
     - In this demo, complaints are simply printed to the console.

Usage Instructions:
1. **Clone the repository**  
   ```
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**  
   Make sure you have a compatible Python environment. Install the required packages:
   ```
   pip install pandas seaborn matplotlib scikit-learn faker cryptography numpy
   ```

3. **Run the Demo**  
   Execute the main script:
   ```
   python blockchain.py
   ```
   - This will generate synthetic datasets for multiple clients, train local logistic regression models, encrypt weights, record transactions on the blockchain, aggregate models, evaluate performance, handle complaints, and output files.


4. **Output Files**  
   - **Console Output**: Includes:
     - Encrypted transactions (truncated for readability)
     - Received complaints with timestamps
     - Blockchain blocks with proofs and transactions
     - Final metrics DataFrame showing accuracy, precision, recall, and AUC-ROC for each client.
   - Note: Confusion matrix and ROC curve generation has been removed; metrics are printed to the console.
5. **Modify Parameters**  
   - **num_clients**: Change the number of simulated clients (default is 5).  
   - **num_samples**: Adjust the total number of synthetic samples (must be a multiple of 3).  
   - **noise_factor**: Control label noise intensity during training and evaluation.  
   - **KMeans Clusters**: Modify `n_clusters` in clustering step if needed.  
   - **Logistic Regression Settings**: Customize hyperparameters (e.g., `C`, `max_iter`) when initializing `LogisticRegression()`.  

6. **Extend for Neural Network**  
   - This implementation uses logistic regression for simplicity. To replace with a neural network:
     - Import an appropriate neural network model (e.g., `MLPClassifier` or a custom PyTorch/TensorFlow model).  
     - Replace the `LogisticRegression()` initialization in each client training step with the chosen neural network model.  
     - Adjust encryption to handle larger model weight structures accordingly.

7. **Citation Request**  
   If you use or build upon this code, please cite:
   ```
   Alshareet, O., Awasthi, A. A Novel Framework for Integrating Blockchain-Driven Federated Learning with Neural Networks in E-Commerce. J Netw Syst Manage 33, 56 (2025).
   DOI: https://doi.org/10.1007/s10922-025-09928-x
   ```


