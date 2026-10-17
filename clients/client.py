import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
import collections
import paho.mqtt.client as mqtt

MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_CLASSES = 10
GAMMA_FOCAL = 2.0
DROPOUT_RATE = 0.3
BATCH_SIZE = 512

GLOBAL_ATTACK_NAMES = [
    'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 
    'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms'
]

class MulticlassFocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.clamp(min=1e-7, max=1.0 - 1e-7)
        ce = F.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean() if self.reduction == 'mean' else focal.sum() if self.reduction == 'sum' else focal

class IDS_MLP(nn.Module):
    def __init__(self, input_size, num_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class FederatedClient:
    def __init__(self, client_id, train_path, test_path):
        self.client_id = client_id
        self.active_round = False
        self.msg_payload = None
        self.global_scaler = None
        
        self.target_names = GLOBAL_ATTACK_NAMES
        self.le = LabelEncoder().fit(self.target_names)

        self.extract_raw_data(train_path, test_path)

        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self.on_connect
        self.mqtt.on_message = self.on_message

    def extract_raw_data(self, train_path, test_path):
        print(f"\n[CLIENT {self.client_id}] 📜 Lecture brute des bases de données CSV...")
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        possible_target_cols = ['attack_cat', 'type', 'target', 'label']
        target_col = next((col for col in possible_target_cols if col in train.columns), train.columns[-1])
        print(f"[CLIENT {self.client_id}] 🎯 Variable cible détectée : '{target_col}'")

        train_labels = train[target_col].astype(str).str.strip()
        test_labels = test[target_col].astype(str).str.strip()

        if train_labels.str.isnumeric().all():
            self.y_tr = train_labels.astype(np.int64).values
            self.y_te = test_labels.astype(np.int64).values
        else:
            self.y_tr = self.le.transform(train_labels).astype(np.int64)
            self.y_te = self.le.transform(test_labels).astype(np.int64)

        drop_cols = ['id', 'label', 'target', 'ts', 'attack_cat', 'type']
        X_tr_df = train.drop(columns=[c for c in drop_cols if c in train.columns])
        X_te_df = test.drop(columns=[c for c in drop_cols if c in test.columns])

        for col in ['proto', 'service', 'state']:
            if col in X_tr_df.columns:
                le_feat = LabelEncoder()
                all_v = pd.concat([X_tr_df[col], X_te_df[col]]).astype(str)
                le_feat.fit(all_v)
                X_tr_df[col] = le_feat.transform(X_tr_df[col].astype(str))
                X_te_df[col] = le_feat.transform(X_te_df[col].astype(str))

        self.X_tr_raw = X_tr_df.values.astype(np.float32)
        self.X_te_raw = X_te_df.values.astype(np.float32)
        self.input_size = self.X_tr_raw.shape[1]
    
    def process_and_build_loaders(self):
        print(f"[CLIENT {self.client_id}] 🧊 Alignement spatial via Federated Normalization Scaler...")
        X_tr = (self.X_tr_raw - self.global_scaler["mean"]) / self.global_scaler["std"]
        X_te = (self.X_te_raw - self.global_scaler["mean"]) / self.global_scaler["std"]

        print(f"[CLIENT {self.client_id}] 📊 Calcul des poids de classes sur les données réelles...")
        counts_arr = np.bincount(self.y_tr, minlength=NUM_CLASSES).astype(np.float32)
        counts_arr = np.where(counts_arr == 0, 1e9, counts_arr)
        w = 1.0 / np.sqrt(counts_arr)
        w[counts_arr > 1e8] = 0.0
        self.class_weights = torch.tensor(w, dtype=torch.float32)

        sample_w = torch.tensor([w[y] for y in self.y_tr], dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

        print(f"[CLIENT {self.client_id}] 📦 Création des PyTorch DataLoaders...")
        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(self.y_tr, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(self.y_te, dtype=torch.long))

        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        self.test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        self.model = IDS_MLP(input_size=self.input_size, num_classes=NUM_CLASSES, dropout=DROPOUT_RATE)
        self.loss_fn = MulticlassFocalLoss(weights=self.class_weights, gamma=GAMMA_FOCAL)
        self.opt = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        
        print(f"[CLIENT {self.client_id}] ✅ Configuration terminée avec succès.")
        print(f"[CLIENT {self.client_id}] ⚡ Notification de synchronisation envoyée au serveur. En attente du Round 1...")
        self.mqtt.publish(f"client/ready/{self.client_id}", b"ready")

    def train(self, epochs=3):
        self.model.train()
        global_weights = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
        mu = 0.1  

        for epoch in range(epochs):
            total_loss, n_samples = 0.0, 0
            for X, y in self.train_loader:
                self.opt.zero_grad()
                out = self.model(X)
                
                base_loss = self.loss_fn(out, y)
                
                prox_term = 0.0
                for name, param in self.model.named_parameters():
                    if name in global_weights:
                        prox_term += torch.sum((param - global_weights[name]) ** 2)
                
                loss = base_loss + (mu / 2.0) * prox_term
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()
                
                total_loss += loss.item() * X.size(0)
                n_samples += X.size(0)
            print(f"   [Époque {epoch+1}/{epochs}] Perte composite (Focal + FedProx) : {total_loss / n_samples:.5f}")

    def evaluate(self):
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in self.test_loader:
                preds = torch.argmax(self.model(X), dim=1)
                all_preds.extend(preds.numpy())
                all_targets.extend(y.numpy())

        all_preds, all_targets = np.array(all_preds), np.array(all_targets)
        unique = sorted(np.unique(np.concatenate([all_targets, all_preds])).tolist())
        names = [self.target_names[i] if i < len(self.target_names) else f"Class_{i}" for i in unique]

        report = classification_report(all_targets, all_preds, labels=unique, target_names=names, output_dict=True, zero_division=0)
        acc = accuracy_score(all_targets, all_preds)
        f1_mac = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        print(f"\n[CLIENT {self.client_id}] 📊 RAPPORT DE PERFORMANCE PAR ATTAQUE (LOCAL) :")
        print(f"  Accuracy     : {acc:.4f} | F1 Macro : {f1_mac:.4f}")
        print(f"{'─'*75}")
        print(f"  {'Type d\'Attaque':<20} {'Précision':>10} {'Rappel':>10} {'F1-Score':>10} {'Support':>10}")
        print(f"{'─'*75}")
        for name in names:
            m = report[name]
            print(f"  {name:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {int(m['support']):>10}")
        print(f"{'─'*75}\n")

        classify_metrics = {}
        for name in names:
            classify_metrics[name.lower()] = {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"]
            }
        return {"accuracy": acc, "f1_macro": f1_mac, "classify_metrics": classify_metrics}

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"[CLIENT {self.client_id}] Connecté. Envoi immédiat des signatures statistiques locales au serveur...")
        client.subscribe("server/global_scaler")
        client.subscribe("server/global")
        
        local_mean = np.mean(self.X_tr_raw, axis=0)
        local_var = np.var(self.X_tr_raw, axis=0)
        scaler_payload = pickle.dumps({"mean": local_mean, "var": local_var, "n": len(self.X_tr_raw)})
        client.publish(f"client/scaler_stats/{self.client_id}", scaler_payload, retain=False) # Retain à False pour éviter de polluer le broker

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload

        if len(payload) == 0:
            return

        if topic == "server/global_scaler" and self.global_scaler is None:
            self.global_scaler = pickle.loads(payload)
            self.process_and_build_loaders()
        elif topic == "server/global" and self.global_scaler is not None:
            self.msg_payload = payload
            self.active_round = True

    def start(self):
        self.mqtt.connect(MQTT_BROKER, MQTT_PORT)
        self.mqtt.loop_start()

        try:
            while True:
                if self.active_round and self.msg_payload is not None:
                    self.model.load_state_dict(pickle.loads(self.msg_payload))
                    print(f"\n[CLIENT {self.client_id}] ── NOUVEAU ROUND FL (3 ÉPOQUES LOCALES + PROTECTION FEDPROX) ──")
                    
                    self.train(epochs=3)
                    metrics = self.evaluate()

                    response_payload = pickle.dumps({
                        "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "classify_metrics": metrics["classify_metrics"]
                    })
                    self.mqtt.publish(f"client/update/{self.client_id}", response_payload)
                    
                    self.active_round = False
                    self.msg_payload = None
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.mqtt.loop_stop()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 client_fl_multiclass.py <client_id> <train_path> <test_path>")
        sys.exit(1)
    c = FederatedClient(sys.argv[1], sys.argv[2], sys.argv[3])
    c.start()