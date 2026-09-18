import uvicorn
import pickle
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from model.server import Server
from model.client import Client
from sklearn.utils import shuffle
import pandas as pd
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import requests
import ast
import random
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np
tf.get_logger().setLevel('ERROR')
try:
    import tensorflow_privacy
    from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
    _TF_PRIVACY_AVAILABLE = True
except ImportError:
    # Optional: only needed for method == "differential privacy".
    # tf-privacy 0.9.0 has no Python 3.12 wheel (requires <3.12), so on
    # newer interpreters (e.g. Kaggle) non-DP paths still work.
    tensorflow_privacy = None
    _TF_PRIVACY_AVAILABLE = False


server = ''
X_test_list = []
y_test_list = []
clients_datalist_X = []
clients_datalist_y = []

class Input(BaseModel):
    s_hash: list[str]
    x: int

class User_Input():
    client: int
    committee: int
    threshold: int

# Initializing the fast API server
app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


max_timestamp = 35
current_timestamp = 0
dataset = ""
method = ""

# Adversarial-injection hooks (default OFF = exact original behavior).
# run_experiment.py sets these for Phase D robustness runs.
ADVERSARIAL_IDS = set()
ATTACK = None  # None | "scaling"
ATTACK_SCALE = 5.0


def _maybe_apply_scaling_attack(client_model, client_id):
    if ATTACK == "scaling" and client_id in ADVERSARIAL_IDS:
        client_model.set_weights([w * ATTACK_SCALE for w in client_model.get_weights()])

epochs = 5
batch_size = 64

l2_norm_clip = 1.0
noise_multiplier = 0.3
num_microbatches = 1
learning_rate = 0.25

def get_dp_optimizer():
    if not _TF_PRIVACY_AVAILABLE:
        raise RuntimeError(
            "tensorflow_privacy is not installed in this environment "
            "(its latest release supports Python <3.12 only); "
            "method='differential privacy' is unavailable here.")
    return tensorflow_privacy.DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate)

optimizer = None  # built lazily via get_dp_optimizer() (DP branch only)

loss_categorical = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss_binary = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def get_dummy(X, y, n):
    unique_element = []
    unique_element_index = []

    if dataset == 'minst':
        for i in range(len(X)):
            for j in range(n):
                if y[i][j] == 1 and j not in unique_element:
                    unique_element.append(j)
                    unique_element_index.append(i)
    else:
        for i in range(y.shape[0]):
            if y[i] not in unique_element:
                unique_element.append(y[i])
                unique_element_index.append(i)

    y_dummy = y[unique_element_index]
    X_dummy = X[unique_element_index]
    return np.array(X_dummy), np.array(y_dummy)

X_test = 0
y_test = 0

from collections import Counter
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import random

def minst_dataset(no_of_client):
    global X_test, y_test
    (train_images, train_labels), (X_test, y_test) = mnist.load_data()

    train_images = train_images.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    y_test = to_categorical(y_test)

    X_dummy, y_dummy = get_dummy(train_images, train_labels, 10)

    _X, _y = {}, {}
    for i in range(10):
        _X[i] = []
        _y[i] = []
    
    for i in range(train_labels.shape[0]):
        for j in range(train_labels.shape[1]):
            if train_labels[i][j]:
                _X[j].append(train_images[i])
                _y[j].append(train_labels[i])

    for i in range(no_of_client):          
        iterative_list_X = {}
        iterative_list_y = {}
        for j in range(max_timestamp):
            iterative_list_X[j] = []
            iterative_list_y[j] = []
        
        for t in range(10):           
            client_X = _X[t][i*len(_X[t])//no_of_client : (i+1)*len(_X[t])//no_of_client]
            client_y = _y[t][i*len(_y[t])//no_of_client : (i+1)*len(_y[t])//no_of_client]
            
            for j in range(max_timestamp):
                iterative_list_X[j] += client_X[j*len(client_X)//max_timestamp : (j+1)*len(client_X)//max_timestamp]
                iterative_list_y[j] += client_y[j*len(client_y)//max_timestamp : (j+1)*len(client_y)//max_timestamp]

        for j in range(max_timestamp):
            combined_data = list(zip(iterative_list_X[j], iterative_list_y[j]))
            random.shuffle(combined_data)
            iterative_list_X[j], iterative_list_y[j] = zip(*combined_data)
        
        clients_datalist_X.append(iterative_list_X)
        clients_datalist_y.append(iterative_list_y)
    
    for i in range(no_of_client):
        for j in range(max_timestamp):
            clients_datalist_X[i][j] = np.concatenate((np.array(clients_datalist_X[i][j]), X_dummy), axis=0)
            clients_datalist_y[i][j] = np.concatenate((np.array(clients_datalist_y[i][j]), y_dummy), axis=0)


def credit_card_fraud_detection_dataset(no_of_client):
    global X_test, y_test

    data = pd.read_csv('model/creditcard_2023.csv')
    X = np.array(data.drop(['id', 'Class'], axis=1))
    y = np.array(data['Class'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

    X_dummy, y_dummy = get_dummy(X_train, y_train, 2)
    _X, _y = {}, {}
    for i in range(2):
        _X[i] = []
        _y[i] = []

    for i in range(y_train.shape[0]):
        _X[int(y_train[i])].append(X_train[i])
        _y[int(y_train[i])].append(y_train[i])

    for i in range(no_of_client):
        iterative_list_X = {}
        iterative_list_y = {}
        for j in range(max_timestamp):
            iterative_list_X[j] = []
            iterative_list_y[j] = []

        for t in range(2):
            client_X = _X[t][i*len(_X[t])//no_of_client : (i+1)*len(_X[t])//no_of_client]
            client_y = _y[t][i*len(_y[t])//no_of_client : (i+1)*len(_y[t])//no_of_client]

            for j in range(max_timestamp):
                iterative_list_X[j] += client_X[j*len(client_X)//max_timestamp : (j+1)*len(client_X)//max_timestamp]
                iterative_list_y[j] += client_y[j*len(client_y)//max_timestamp : (j+1)*len(client_y)//max_timestamp]

        for j in range(max_timestamp):
            combined_data = list(zip(iterative_list_X[j], iterative_list_y[j]))
            random.shuffle(combined_data)
            iterative_list_X[j], iterative_list_y[j] = zip(*combined_data)

        clients_datalist_X.append(iterative_list_X)
        clients_datalist_y.append(iterative_list_y)

    for i in range(no_of_client):
        for j in range(max_timestamp):
            clients_datalist_X[i][j] = np.concatenate((np.array(clients_datalist_X[i][j]), X_dummy), axis=0)
            clients_datalist_y[i][j] = np.concatenate((np.array(clients_datalist_y[i][j]), y_dummy), axis=0)


def nbaiot_dataset(no_of_client, data_root='model/nbaiot', per_device=30000, seed=42, alpha=None):
    """N-BaIoT split (guideline 2.1): natural device->client mapping when
    no_of_client <= n_devices, else Dirichlet(alpha) split of pooled data.
    Same per-timestamp dict structure + dummy-sample convention as the
    minst/credit-card loaders above."""
    global X_test, y_test
    from nbaiot_loader import load_nbaiot
    devices = load_nbaiot(data_root, per_device=per_device, seed=seed)
    names = sorted(devices.keys())

    shards_X, shards_y = [], []
    if no_of_client <= len(names):
        for n in names[:no_of_client]:
            Xd, yd = devices[n]
            shards_X.append(Xd)
            shards_y.append(yd)
        # leftover devices' data is unused in this run (natural-split pilot)
    else:
        if alpha is None:
            alpha = 100
        from dirichlet_partition import dirichlet_partition
        Xp = np.concatenate([devices[n][0] for n in names], axis=0)
        yp = np.concatenate([devices[n][1] for n in names], axis=0)
        parts = dirichlet_partition(yp, no_of_client, alpha, seed=seed)
        for p in parts:
            shards_X.append(Xp[p] if len(p) else np.zeros((0, Xp.shape[1]), dtype=np.float32))
            shards_y.append(yp[p] if len(p) else np.zeros((0,), dtype=np.int64))

    # global test set: 15% stratified holdout from pooled train data
    Xp_all = np.concatenate([s for s in shards_X if len(s)], axis=0)
    yp_all = np.concatenate([s for s in shards_y if len(s)], axis=0)
    X_tr, X_test, y_tr, y_test = train_test_split(
        Xp_all, yp_all, test_size=0.15, random_state=seed, stratify=yp_all)
    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_test = scaler.transform(X_test)
    shards_X = [scaler.transform(s) if len(s) else s for s in shards_X]

    for i in range(no_of_client):
        Xc, yc = shards_X[i], shards_y[i]
        if len(Xc) == 0:
            # empty Dirichlet shard: reuse one sample per class from pool so
            # client code paths (fit/evaluate) never see zero-size input
            for c in np.unique(yp_all):
                j = np.where(yp_all == c)[0][0]
                Xc = np.concatenate([Xc, scaler.transform(Xp_all[j:j + 1])], axis=0)
                yc = np.concatenate([yc, yp_all[j:j + 1]], axis=0)
        X_dummy, y_dummy = get_dummy(Xc, yc, 2)
        order = np.arange(len(Xc))
        random.Random(seed + i).shuffle(order)
        Xc, yc = Xc[order], yc[order]
        chunks_X = np.array_split(Xc, max_timestamp)
        chunks_y = np.array_split(yc, max_timestamp)
        iterative_list_X, iterative_list_y = {}, {}
        for j in range(max_timestamp):
            iterative_list_X[j] = np.concatenate([np.asarray(chunks_X[j]), X_dummy], axis=0)
            iterative_list_y[j] = np.concatenate([np.asarray(chunks_y[j]), y_dummy], axis=0)
        clients_datalist_X.append(iterative_list_X)
        clients_datalist_y.append(iterative_list_y)


def split_dataset_between_clients(no_of_client):
    if dataset == "minst":
        minst_dataset(no_of_client)
    elif dataset == 'credit card':
        credit_card_fraud_detection_dataset(no_of_client)
    elif dataset == 'nbaiot':
        nbaiot_dataset(no_of_client)


def get_model():
    model = ''
    if dataset == 'minst':
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
    elif dataset == 'credit card':
            model = Sequential([
            Flatten(input_shape=(29,)),  
            Dense(128, activation='relu'),  
            Dense(1, activation='sigmoid')  
        ])
    elif dataset == 'nbaiot':
        # N-BaIoT: 115 statistical traffic features, binary benign/attack
        model = Sequential([
            Flatten(input_shape=(115,)),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    else:
        pass
    if method != "differential privacy":
        if dataset != 'minst':
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        if dataset == 'minst':
            model.compile(optimizer=get_dp_optimizer(), loss=loss_categorical, metrics=['accuracy'])
        else:
            model.compile(optimizer=get_dp_optimizer(), loss=loss_binary, metrics=['accuracy'])
    return model


def model_aggregation(model_list):
    model = get_model()

    aggregate_weights = model_list[0]

    for i in range(1, len(model_list)):
        for j, w in enumerate(model_list[i]):
            aggregate_weights[j] += w

    for j, w in enumerate(aggregate_weights):
        aggregate_weights[j] /= len(model_list)

    model.set_weights(aggregate_weights)
    return model
    
def train_threshold_times(client, committee, threshold):
    trust_score_list = {}
    average_trust_score_list = []
    weights_list = []
    global current_timestamp
    for i in range(client):
        trust_score_list[i] = []
    for i in range(threshold):
        committee_memeber = [random.randint(0, client-1) for _ in range(committee)]
        for j in range(client):
            if j in committee_memeber:
                trust_score_list[j].append(0)
                continue
            
            model = get_model()
            model.fit(clients_datalist_X[i][current_timestamp], clients_datalist_y[i][current_timestamp], epochs=epochs, batch_size=batch_size, validation_split=0.2)
            weights = model.get_weights()
            weights_list.append(weights)

            accuracy_sum = 0
            for k in committee_memeber:
                score = model.evaluate(x=clients_datalist_X[k][current_timestamp], y=clients_datalist_y[k][current_timestamp], verbose=0)
                accuracy = score[1]
                accuracy_sum += accuracy
            trust_score_list[j].append(accuracy_sum/committee)

        current_timestamp += 1

    for i in range(client):
        average_trust_score = sum(trust_score_list[i])/ len(trust_score_list[i])
        average_trust_score_list.append(average_trust_score)
    
    aggregated_model = model_aggregation(weights_list)

    return average_trust_score_list, aggregated_model

    
ttt = ''

def update_server_model(model):
    # Local content-addressed storage (replaces defunct Infura IPFS, see 5.2).
    # Returns a hex hash string — same type/shape as before, still written
    # on-chain via setServer downstream.
    from local_storage import save_weights

    weights = model.get_weights()

    server.model.set_weights(weights)

    server_grads = list()

    server_grads.append(weights)

    server_file = open("server.txt", "w+")

    server_content = str(server_grads)

    server_file.write(server_content)
    server_file.close()

    server_hash = save_weights(weights)

    global ttt
    ttt = server_hash
    return server_hash


@app.post("/initialize/")
async def initialize(request: Request):
    clients_datalist_X.clear()
    clients_datalist_y.clear()
    global server
    global dataset
    global method

    # Retrieve JSON data from the HTTP request
    res = await request.json()

    dataset = res["dataset"]
    method = res["method"]
    server = Server(dataset, method)

    # Split the dataset between clients
    split_dataset_between_clients(res["client"])

    # Train the model and calculate trust scores
    trust_score, aggregated_model = train_threshold_times(res["client"], res["committee"], res["threshold"])
    
    # Convert trust scores to integer format for better representation
    for i in range(len(trust_score)):
        trust_score[i] = int(trust_score[i] * 100000)
    
    # Update the server model and obtain the hash
    hash = update_server_model(aggregated_model)

    return {"trust_score": trust_score,
            "server_hash": hash}


from requests.exceptions import ChunkedEncodingError

def copy_server(hash):
    # Local content-addressed storage (replaces defunct Infura IPFS, see 5.2).
    from local_storage import load_weights
    global ttt

    print("Before Hash: ", ttt)
    print("After Hash : ", hash)

    retrieved_weights = load_weights(hash)

    server.model.set_weights(retrieved_weights)


def calculate_new_model(model, index, length, client):
    aggregated_model = get_model()
    weight = []

    for i, w in enumerate(model.get_weights()):
        new_weight = ((length-index)*w - client.model.get_weights()[i])/(length-index-1)
        weight.append(new_weight)
    aggregated_model.set_weights(weight)
    
    return aggregated_model

def get_trust_model(res):
    client_list = list()
    for i in range(res["client"]):                   
        temp = Client(i, dataset, method)
        temp.model.set_weights(server.model.get_weights())
        client_list.append(temp)          

    committee_memeber = sorted(range(len(res["trust_score"])), key=lambda i: res["trust_score"][i], reverse=True)[:3]

    model_list = []
    for i in range(res["client"]):
        if i in committee_memeber:
            continue
        client_list[i].model.fit(clients_datalist_X[i][current_timestamp], clients_datalist_y[i][current_timestamp], epochs=epochs, batch_size=batch_size, validation_split=0.2)
        _maybe_apply_scaling_attack(client_list[i].model, i)
        model_list.append(client_list[i].model.get_weights())

    aggregated_model = model_aggregation(model_list)
    prev_score = 0
    curr_score = 0
    for k in committee_memeber:
        score = aggregated_model.evaluate(x=clients_datalist_X[k][current_timestamp], y=clients_datalist_y[k][current_timestamp], verbose=0)
        accuracy = score[1]
        curr_score += accuracy
        
        score = server.model.evaluate(x=clients_datalist_X[k][current_timestamp], y=clients_datalist_y[k][current_timestamp], verbose=0)
        accuracy = score[1]
        prev_score += accuracy
    curr_score = int(curr_score/res["committee"] * 100000)
    prev_score = int(prev_score/res["committee"] * 100000)
    
    updated_trust_score = []
    for score in res["trust_score"]:
        updated_trust_score.append(int(score))
    for i in committee_memeber:
        updated_trust_score[i] = max(int(res["trust_score"][i])-5000, 0)
    non_committee_member = [i for i in range(len(res["trust_score"])) if i not in committee_memeber]
    if curr_score > prev_score:
        top_50_per_scorer = sorted(non_committee_member, key=lambda i: res["trust_score"][i])[:len(non_committee_member)//2]
        for i in non_committee_member:
            if i in top_50_per_scorer:
                updated_trust_score[i] = max(int(res["trust_score"][i])-5000, 0)
            else:
                accuracy_sum = 0
                for k in committee_memeber:
                    score = client_list[i].model.evaluate(x=clients_datalist_X[k][current_timestamp], y=clients_datalist_y[k][current_timestamp], verbose=0)
                    accuracy = score[1]
                    accuracy_sum += accuracy
                updated_trust_score[i] = int(accuracy_sum/res["committee"] * 100000)
        hash = update_server_model(aggregated_model)
    else:
        non_committee_trust_scores = [updated_trust_score[x] for x in non_committee_member]
        combined = zip(non_committee_trust_scores, non_committee_member)
        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=False)
        non_committee_trust_scores = [x[0] for x in sorted_combined]
        non_committee_member = [x[1] for x in sorted_combined]
        
        index = 0
        global_aggregated_model = server.model
        new_aggregated_model = aggregated_model
        while curr_score < prev_score and index < len(non_committee_member):
            updated_trust_score[non_committee_member[index]] = int(updated_trust_score[non_committee_member[index]])
            curr_score = 0
            new_aggregated_model = calculate_new_model(new_aggregated_model, index, len(non_committee_member), client_list[non_committee_member[index]])
            for k in committee_memeber:
                score = new_aggregated_model.evaluate(x=clients_datalist_X[k][current_timestamp], y=clients_datalist_y[k][current_timestamp], verbose=0)
                accuracy = score[1]
                curr_score += accuracy
            curr_score = int(curr_score/res["committee"] * 100000)
            index += 1
        if index == len(non_committee_member):
            new_aggregated_model = global_aggregated_model
        temp = [non_committee_member[i] for i in range(index, len(non_committee_member))]
        top_50_per_scorer = sorted(temp, key=lambda i: res["trust_score"][i])[:len(temp)//2]
        for i in range(index, len(non_committee_member)):
            if i in top_50_per_scorer:
                updated_trust_score[i] = max(int(res["trust_score"][i])-5000, 0)
            else:
                accuracy_sum = 0
                for k in committee_memeber:
                    score = client_list[i].model.evaluate(x=clients_datalist_X[k][current_timestamp], y=clients_datalist_y[k][current_timestamp], verbose=0)
                    accuracy = score[1]
                    accuracy_sum += accuracy
                updated_trust_score[i] = int(accuracy_sum/res["committee"] * 100000)
        hash = update_server_model(new_aggregated_model)
    
    return hash, updated_trust_score
    
def get_committee_consensus_model(res):
    client_list = list()
    for i in range(res["client"]):
        temp = Client(i, dataset, method)
        temp.model.set_weights(server.model.get_weights())
        client_list.append(temp)

    committee_memeber = sorted(range(len(res["trust_score"])), key=lambda i: res["trust_score"][i], reverse=True)[:res["committee"]]

    trust_score = res["trust_score"]
    for i in range(res["client"]):
        if i in committee_memeber:
            trust_score[i] = max(int(res["trust_score"][i])-5000, 0)
            continue
        client_list[i].model.fit(clients_datalist_X[i][current_timestamp], clients_datalist_y[i][current_timestamp], epochs=epochs, batch_size=batch_size, validation_split=0.2)
        _maybe_apply_scaling_attack(client_list[i].model, i)
        curr_score = 0
        for k in committee_memeber:
            sc = client_list[i].model.evaluate(x=clients_datalist_X[k][current_timestamp], y=clients_datalist_y[k][current_timestamp], verbose=0)
            accuracy = sc[1]
            curr_score += accuracy
        curr_score = int(curr_score/res["committee"] * 100000)
        trust_score[i] = curr_score

    non_committee_member = [i for i in range(res["client"]) if i not in committee_memeber]
    non_committee_trust_scores = [trust_score[x] for x in non_committee_member]

    def get_top_50_percent_indices(lst):
        sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
        top_50_percent_index = len(lst) // 2
        return sorted_indices[:top_50_percent_index]

    max_50_per_index = get_top_50_percent_indices(non_committee_trust_scores)

    weights_list = []
    for i in max_50_per_index:
        weights_list.append(client_list[i].model.get_weights())

    aggregated_model = model_aggregation(weights_list)

    prev_score = 0
    curr_score = 0
    for k in committee_memeber:
        score = aggregated_model.evaluate(x=clients_datalist_X[k][current_timestamp], y=clients_datalist_y[k][current_timestamp], verbose=0)
        accuracy = score[1]
        curr_score += accuracy

        score = server.model.evaluate(x=clients_datalist_X[k][current_timestamp], y=clients_datalist_y[k][current_timestamp], verbose=0)
        accuracy = score[1]
        prev_score += accuracy
    curr_score = int(curr_score/res["committee"] * 100000)
    prev_score = int(prev_score/res["committee"] * 100000)

    if curr_score > prev_score:
        hash = update_server_model(aggregated_model)
    else:
        hash = update_server_model(server.model)

    return hash, trust_score


@app.post("/trust_model/")
async def trust_model(request:Request):
    res = await request.json()
    global current_timestamp

    copy_server(res["hash"])
    
    hash = ''
    updated_trust_score = []

    if method != "committee consensus":
        hash, updated_trust_score = get_trust_model(res)
    else:
        hash, updated_trust_score = get_committee_consensus_model(res)
    
    current_timestamp += 1

    return {"server_hash": hash,
            "trust_score": updated_trust_score}


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
@app.post("/model_validation/")
async def model_validation(request: Request):
    res = await request.json()
    copy_server(res["server_hash"])
    global X_test, y_test

    probabilities = server.model.predict(X_test)

    if dataset == 'minst':
        predicted_labels = np.argmax(probabilities, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        y_true = true_labels
    else:
        predicted_labels = np.round(probabilities)
        y_true = y_test

    y_pred = predicted_labels

    print(classification_report(y_true, y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    
    with open('accuracy.txt', 'a') as f:
        value = str(accuracy) + " "  
        f.write(value)

    precision = precision_score(y_true, y_pred, average='weighted')
    print("Precision:", precision)

    with open('precision.txt', 'a') as f:
        value = str(precision) + " "  
        f.write(value)

    recall = recall_score(y_true, y_pred, average='weighted')
    print("Recall:", recall)

    with open('recall.txt', 'a') as f:
        value = str(recall) + " "  
        f.write(value)

    f1 = f1_score(y_true, y_pred, average='weighted')
    print("F1-score:", f1)

    with open('f1.txt', 'a') as f:
        value = str(f1) + " "  
        f.write(value)

    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1}

# Configuring the server host and port
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')