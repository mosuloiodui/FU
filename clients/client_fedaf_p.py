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
GENERIC_INDEX = GLOBAL_ATTACK_NAMES.index('Generic')  # = 5

# ══════════════════════════════════════════════════════════════════
#  COUCHE 1 — PÉNALITÉ D'ALIGNEMENT (ALIGNMENT PENALTY)
#  Empêche AdamW de reconstruire les features de 'Generic'
#  dans les couches cachées lors de l'entraînement de routine.
# ══════════════════════════════════════════════════════════════════
LAMBDA_UNLEARN_INIT = 0.15   # Force initiale de la pénalité
LAMBDA_UNLEARN_DECAY = 0.97  # Décroissance par round (stabilisation progressive)
LAMBDA_UNLEARN_FLOOR = 0.02  # Plancher minimum (ne jamais désactiver totalement)

def compute_alignment_penalty(model: nn.Module,
                               unlearn_signal: dict,
                               lambda_u: float) -> torch.Tensor:
    """
    Calcule la pénalité cosinus entre le gradient courant de chaque
    paramètre et la direction d'oubli correspondante.

    Pénalise uniquement les composantes qui pointent DANS la direction
    interdite (relu sur la similarité cosinus → pas de pénalité inverse).
    """
    penalty = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if name not in unlearn_signal or param.grad is None:
            continue
        u_dir   = unlearn_signal[name].float().to(param.device).flatten()
        g_flat  = param.grad.float().flatten()
        if u_dir.norm() < 1e-9 or g_flat.norm() < 1e-9:
            continue
        cos_sim = F.cosine_similarity(g_flat.unsqueeze(0),
                                      u_dir.unsqueeze(0))
        penalty = penalty + lambda_u * torch.relu(cos_sim)
    return penalty


# ══════════════════════════════════════════════════════════════════
#  MODÈLES
# ══════════════════════════════════════════════════════════════════
class MulticlassFocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weights   = weights
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs  = F.softmax(inputs, dim=1)
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-7, 1.0 - 1e-7)
        ce     = F.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
        focal  = (1.0 - pt) ** self.gamma * ce
        if self.reduction == 'mean':   return focal.mean()
        if self.reduction == 'sum':    return focal.sum()
        return focal


class IDS_MLP(nn.Module):
    def __init__(self, input_size, num_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════
#  CLIENT FEDAF PERSISTANT
# ══════════════════════════════════════════════════════════════════
class FederatedClient:
    def __init__(self, client_id, train_path, test_path):
        self.client_id     = str(client_id)
        self.active_round  = False
        self.msg_payload   = None
        self.global_scaler = None
        self.current_round = 1

        # ── État de la protection persistante (Couche 1) ──
        self.unlearn_signal  = None   # Direction d'oubli reçue du serveur
        self.lambda_unlearn  = LAMBDA_UNLEARN_INIT
        self.unlearning_done = False  # Passe à True après le Round 1 d'oubli

        self.target_names = GLOBAL_ATTACK_NAMES
        self.le = LabelEncoder().fit(self.target_names)
        self.extract_raw_data(train_path, test_path)

        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self.on_connect
        self.mqtt.on_message  = self.on_message

    # ──────────────────────────────────────────────────────────────
    def extract_raw_data(self, train_path, test_path):
        print(f"\n[CLIENT {self.client_id}] 📜 Lecture brute des CSV...")
        train = pd.read_csv(train_path).dropna()
        test  = pd.read_csv(test_path).dropna()

        possible = ['attack_cat', 'type', 'target', 'label']
        target_col = next((c for c in possible if c in train.columns), train.columns[-1])

        tr_lbl = train[target_col].astype(str).str.strip()
        te_lbl = test[target_col].astype(str).str.strip()

        if tr_lbl.str.isnumeric().all():
            self.y_tr = tr_lbl.astype(np.int64).values
            self.y_te = te_lbl.astype(np.int64).values
        else:
            self.y_tr = self.le.transform(tr_lbl).astype(np.int64)
            self.y_te = self.le.transform(te_lbl).astype(np.int64)

        drop_cols = ['id', 'label', 'target', 'ts', 'attack_cat', 'type']
        X_tr_df = train.drop(columns=[c for c in drop_cols if c in train.columns])
        X_te_df = test.drop(columns=[c for c in drop_cols if c in test.columns])

        for col in ['proto', 'service', 'state']:
            if col in X_tr_df.columns:
                le_f = LabelEncoder()
                le_f.fit(pd.concat([X_tr_df[col], X_te_df[col]]).astype(str))
                X_tr_df[col] = le_f.transform(X_tr_df[col].astype(str))
                X_te_df[col] = le_f.transform(X_te_df[col].astype(str))

        self.X_tr_raw  = X_tr_df.values.astype(np.float32)
        self.X_te_raw  = X_te_df.values.astype(np.float32)
        self.input_size = self.X_tr_raw.shape[1]

    # ──────────────────────────────────────────────────────────────
    def process_and_build_loaders(self):
        print(f"[CLIENT {self.client_id}] 🧊 Normalisation fédérée...")
        mean = self.global_scaler["mean"]
        std  = self.global_scaler["std"]
        self.X_tr = (self.X_tr_raw - mean) / std
        X_te      = (self.X_te_raw - mean) / std

        counts = np.bincount(self.y_tr, minlength=NUM_CLASSES).astype(np.float32)
        counts = np.where(counts == 0, 1e9, counts)
        w      = 1.0 / np.sqrt(counts)
        w[counts > 1e8] = 0.0
        self.class_weights = torch.tensor(w, dtype=torch.float32)

        sample_w = torch.tensor([w[y] for y in self.y_tr], dtype=torch.float32)
        sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

        train_ds = TensorDataset(torch.tensor(self.X_tr), torch.tensor(self.y_tr, dtype=torch.long))
        test_ds  = TensorDataset(torch.tensor(X_te),      torch.tensor(self.y_te, dtype=torch.long))
        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        self.test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

        self.model   = IDS_MLP(self.input_size, NUM_CLASSES, DROPOUT_RATE)
        self.loss_fn = MulticlassFocalLoss(weights=self.class_weights, gamma=GAMMA_FOCAL)
        self.opt     = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)

        self.mqtt.publish(f"client/ready/{self.client_id}", b"ready")

    # ══════════════════════════════════════════════════════════════
    #  PHASE D'OUBLI (Round 1 uniquement — Client désapprentisseur)
    # ══════════════════════════════════════════════════════════════
    def train_fedaf(self, teacher_dict: dict):
        """
        Désapprentissage chirurgical FedAF + calcul de la direction d'oubli
        qui sera broadcastée au serveur pour protéger tous les rounds futurs.
        """
        print(f"\n[CLIENT {self.client_id}] 🧠 FedAF Unlearning — Round {self.current_round}...")

        # ── Séparation Retain / Unlearn ──
        is_generic = (self.y_tr == GENERIC_INDEX)
        X_unlearn  = self.X_tr[is_generic];  y_unlearn = self.y_tr[is_generic]
        X_retain   = self.X_tr[~is_generic]; y_retain  = self.y_tr[~is_generic]

        loader_unlearn = DataLoader(
            TensorDataset(torch.tensor(X_unlearn), torch.tensor(y_unlearn, dtype=torch.long)),
            batch_size=BATCH_SIZE, shuffle=True)
        loader_retain = DataLoader(
            TensorDataset(torch.tensor(X_retain), torch.tensor(y_retain, dtype=torch.long)),
            batch_size=BATCH_SIZE, shuffle=True)

        # ── Teacher immuable ──
        teacher = IDS_MLP(self.input_size, NUM_CLASSES).eval()
        teacher.load_state_dict(teacher_dict)

        # ── Fisher EWC sur les classes saines ──
        self.model.eval()
        ewc_fisher = {n: torch.zeros_like(p)
                      for n, p in self.model.named_parameters() if p.requires_grad}
        crit_ewc = nn.CrossEntropyLoss()
        for X_b, y_b in loader_retain:
            self.model.zero_grad()
            crit_ewc(self.model(X_b), y_b).backward()
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        ewc_fisher[n] += (p.grad.data ** 2) * X_b.size(0)
        for n in ewc_fisher:
            ewc_fisher[n] /= len(X_retain)
        anchor_w = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

        # ── Boucle d'oubli actif (Distillation + EWC + Floutage intermittent) ──
        EPOCHS_FA = 3
        GAMMA_EWC = 35.0
        T         = 2.0
        opt_fa    = optim.AdamW(self.model.parameters(), lr=3e-5)
        self.model.train()

        for epoch in range(EPOCHS_FA):
            iter_unl = iter(loader_unlearn)
            step = 0
            for X_ret, y_ret in loader_retain:
                opt_fa.zero_grad()
                step += 1

                # Distillation rétention
                out_s = self.model(X_ret)
                with torch.no_grad():
                    out_t = teacher(X_ret)
                loss_ret = F.kl_div(F.log_softmax(out_s / T, dim=1),
                                    F.softmax(out_t / T, dim=1),
                                    reduction='batchmean') * (T ** 2)

                # Floutage intermittent de 'Generic' (1 coup sur 3)
                loss_unl = torch.tensor(0.0)
                if step % 3 == 0:
                    try:
                        X_unl, _ = next(iter_unl)
                    except StopIteration:
                        iter_unl = iter(loader_unlearn)
                        X_unl, _ = next(iter_unl)
                    out_unl     = self.model(X_unl)
                    uniform_tgt = torch.full((X_unl.size(0), NUM_CLASSES), 1.0 / NUM_CLASSES)
                    loss_unl    = F.kl_div(F.log_softmax(out_unl / T, dim=1),
                                           uniform_tgt,
                                           reduction='batchmean') * (T ** 2)

                # Terme EWC
                loss_ewc = sum(
                    torch.sum(ewc_fisher[n] * ((p - anchor_w[n]) ** 2))
                    for n, p in self.model.named_parameters() if p.requires_grad
                )

                total = loss_ret + loss_unl + (GAMMA_EWC / 2.0) * loss_ewc
                total.backward()
                opt_fa.step()

        # ── Calcul de la direction d'oubli à envoyer au serveur ──
        # La direction est la somme des gradients de gradient-ascent sur 'Generic'.
        print(f"[CLIENT {self.client_id}] 📐 Calcul de la direction d'oubli (gradient ascent sur 'Generic')...")
        self.model.eval()
        unlearn_direction = {n: torch.zeros_like(p)
                             for n, p in self.model.named_parameters() if p.requires_grad}
        n_batches = 0
        for X_unl, y_unl in loader_unlearn:
            self.model.zero_grad()
            out  = self.model(X_unl)
            # Gradient ascent : maximiser la perte sur 'Generic' = direction de reconstruction
            loss = -F.cross_entropy(out, y_unl)
            loss.backward()
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        unlearn_direction[n] += p.grad.data.clone()
            n_batches += 1
        for n in unlearn_direction:
            if n_batches > 0:
                unlearn_direction[n] /= n_batches

        self.model.zero_grad()
        self.unlearn_signal  = unlearn_direction  # Stocké localement aussi (Couche 1)
        self.unlearning_done = True
        self.lambda_unlearn  = LAMBDA_UNLEARN_INIT

        print(f"[CLIENT {self.client_id}] ✅ Direction d'oubli calculée sur {n_batches} batches.")
        return unlearn_direction

    # ══════════════════════════════════════════════════════════════
    #  ENTRAÎNEMENT DE ROUTINE (FedProx + Alignment Penalty)
    # ══════════════════════════════════════════════════════════════
    def train_normal(self, epochs=3):
        """
        Entraînement standard FedProx.
        Si un signal d'oubli est actif (Couche 1), une pénalité d'alignement
        est ajoutée pour empêcher la reconstruction des features de 'Generic'.
        """
        self.model.train()
        global_weights = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
        mu = 0.1

        use_penalty = (self.unlearn_signal is not None) and self.unlearning_done
        if use_penalty:
            print(f"[CLIENT {self.client_id}] 🛡️  Alignment Penalty active "
                  f"(λ={self.lambda_unlearn:.4f})")

        for epoch in range(epochs):
            for X, y in self.train_loader:
                self.opt.zero_grad()
                out       = self.model(X)
                base_loss = self.loss_fn(out, y)
                prox_term = sum(
                    torch.sum((p - global_weights[n]) ** 2)
                    for n, p in self.model.named_parameters()
                    if n in global_weights
                )
                loss = base_loss + (mu / 2.0) * prox_term
                loss.backward()  # Premier backward pour obtenir les gradients

                if use_penalty:
                    # Calcul de la pénalité d'alignement sur les gradients existants
                    penalty = compute_alignment_penalty(
                        self.model, self.unlearn_signal, self.lambda_unlearn
                    )
                    if penalty.item() > 0:
                        penalty.backward()   # Accumulation sur les gradients existants

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()

        # Décroissance progressive de la pénalité après chaque round
        if use_penalty:
            old = self.lambda_unlearn
            self.lambda_unlearn = max(LAMBDA_UNLEARN_FLOOR,
                                      self.lambda_unlearn * LAMBDA_UNLEARN_DECAY)
            print(f"[CLIENT {self.client_id}] 📉 λ_unlearn : {old:.4f} → {self.lambda_unlearn:.4f}")

    # ──────────────────────────────────────────────────────────────
    def evaluate(self):
        self.model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for X, y in self.test_loader:
                preds_all.extend(torch.argmax(self.model(X), dim=1).numpy())
                targets_all.extend(y.numpy())

        preds_all   = np.array(preds_all)
        targets_all = np.array(targets_all)
        unique = sorted(np.unique(np.concatenate([targets_all, preds_all])).tolist())
        names  = [self.target_names[i] if i < len(self.target_names)
                  else f"Class_{i}" for i in unique]

        report  = classification_report(targets_all, preds_all, labels=unique,
                                         target_names=names, output_dict=True, zero_division=0)
        acc     = accuracy_score(targets_all, preds_all)
        f1_mac  = f1_score(targets_all, preds_all, average='macro', zero_division=0)

        print(f"\n[CLIENT {self.client_id}] 📊 Round {self.current_round} | "
              f"Accuracy: {acc:.4f} | F1 Macro: {f1_mac:.4f}")
        classify_metrics = {}
        for name in names:
            m = report[name]
            classify_metrics[name.lower()] = {
                "precision": m["precision"],
                "recall"   : m["recall"],
                "f1"       : m["f1-score"]
            }
            flag = " ⚠️" if name.lower() == "generic" else ""
            print(f"   • {name:<15} F1:{m['f1-score']:.4f} P:{m['precision']:.4f} "
                  f"R:{m['recall']:.4f}{flag}")

        return {"accuracy": acc, "f1_macro": f1_mac, "classify_metrics": classify_metrics}

    # ──────────────────────────────────────────────────────────────
    def on_connect(self, client, userdata, flags, rc, properties=None):
        client.subscribe("server/global_scaler")
        client.subscribe("server/global")
        client.subscribe("server/round_number")
        client.subscribe("server/unlearn_signal")   # ← Couche 1 : écoute du signal

        local_mean = np.mean(self.X_tr_raw, axis=0)
        local_var  = np.var(self.X_tr_raw,  axis=0)
        client.publish(f"client/scaler_stats/{self.client_id}",
                       pickle.dumps({"mean": local_mean, "var": local_var, "n": len(self.X_tr_raw)}))

    def on_message(self, client, userdata, msg):
        if len(msg.payload) == 0:
            return
        topic = msg.topic

        if topic == "server/global_scaler" and self.global_scaler is None:
            self.global_scaler = pickle.loads(msg.payload)
            self.process_and_build_loaders()

        elif topic == "server/round_number":
            self.current_round = pickle.loads(msg.payload)

        elif topic == "server/unlearn_signal":
            # Réception du signal d'oubli du serveur (broadcasté après Round 1)
            received_signal = pickle.loads(msg.payload)
            if self.unlearn_signal is None:
                self.unlearn_signal  = received_signal
                self.unlearning_done = True
                self.lambda_unlearn  = LAMBDA_UNLEARN_INIT
                print(f"[CLIENT {self.client_id}] 📡 Signal d'oubli reçu du serveur. "
                      f"Alignment Penalty activée (λ={self.lambda_unlearn:.4f}).")

        elif topic == "server/global" and self.global_scaler is not None:
            self.msg_payload  = msg.payload
            self.active_round = True

    # ──────────────────────────────────────────────────────────────
    def start(self):
        self.mqtt.connect(MQTT_BROKER, MQTT_PORT)
        self.mqtt.loop_start()
        try:
            while True:
                if self.active_round and self.msg_payload is not None:
                    received_state = pickle.loads(self.msg_payload)
                    self.model.load_state_dict(received_state)

                    unlearn_direction_to_send = None

                    if self.client_id == "1" and self.current_round == 1 and not self.unlearning_done:
                        # ── Round 1 : Désapprentissage chirurgical (Client 1 uniquement) ──
                        unlearn_direction_to_send = self.train_fedaf(received_state)
                    else:
                        # ── Rounds suivants : FedProx + Alignment Penalty ──
                        self.train_normal(epochs=3)

                    metrics = self.evaluate()

                    # Construction du payload de réponse
                    response = {
                        "state_dict"       : {k: v.cpu() for k, v in self.model.state_dict().items()},
                        "accuracy"         : metrics["accuracy"],
                        "f1_macro"         : metrics["f1_macro"],
                        "classify_metrics" : metrics["classify_metrics"],
                        # ← Couche 2 : direction d'oubli incluse dans la première réponse
                        "unlearn_direction": unlearn_direction_to_send,
                    }
                    self.mqtt.publish(f"client/update/{self.client_id}",
                                      pickle.dumps(response))

                    self.active_round = False
                    self.msg_payload  = None
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.mqtt.loop_stop()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python client.py <client_id> <train_csv> <test_csv>")
        sys.exit(1)
    c = FederatedClient(sys.argv[1], sys.argv[2], sys.argv[3])
    c.start()