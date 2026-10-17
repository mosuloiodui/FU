import sys
import time
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import paho.mqtt.client as mqtt

MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

NUM_CLASSES = 10
TOTAL_ROUNDS = 400
CHECKPOINT_PATH = "global_checkpoint.pth"
SCALER_PATH = "federated_scaler.pkl"
UNLEARN_DIR_PATH = "unlearn_direction.pkl"  # Sauvegarde persistante de la direction d'oubli

GLOBAL_ATTACK_NAMES = [
    'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers',
    'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms'
]
GENERIC_INDEX = GLOBAL_ATTACK_NAMES.index('Generic')  # = 5

# ══════════════════════════════════════════════════════════════════
#  COUCHE 2 — FILTRE D'AGRÉGATION (GRADIENT SURGERY)
#  Projette orthogonalement les deltas clients pour supprimer
#  toute composante qui reconstruirait les features de 'Generic'.
# ══════════════════════════════════════════════════════════════════
class PersistentFedAvg:
    def __init__(self, unlearn_direction: dict, pin_strength_init: float = 0.95):
        """
        unlearn_direction : dict {param_name -> Tensor} — direction d'oubli
                            calculée au Round 1 par le Client désapprentisseur.
        """
        self.pin_strength = pin_strength_init
        self.PIN_DECAY    = 0.92   # Décroissance de 8 % tous les 3 rounds
        self.PIN_FLOOR    = 0.30   # Plancher : jamais en-dessous de 30 %
        self.rounds_since_unlearn = 0

        # Normalisation de la direction d'oubli → vecteurs unitaires par paramètre
        self.forget_subspace = {
            name: F.normalize(grad.float().flatten(), dim=0)
            for name, grad in unlearn_direction.items()
            if grad is not None
        }
        print(f"[SERVER-SURGERY] ✅ Sous-espace d'oubli construit sur "
              f"{len(self.forget_subspace)} tenseurs de paramètres.")

    def _project_out(self, delta: dict) -> dict:
        """
        Pour chaque paramètre p :
            delta_clean = delta - (delta · u) * u
        où u est le vecteur unitaire de la direction d'oubli.
        La composante alignée avec la reconstruction de 'Generic' est supprimée.
        """
        delta_clean = {}
        for name, d in delta.items():
            if name not in self.forget_subspace:
                delta_clean[name] = d
                continue
            u      = self.forget_subspace[name].to(d.device)
            d_flat = d.float().flatten()
            proj   = torch.dot(d_flat, u) * u          # composante interdite
            delta_clean[name] = (d_flat - proj).reshape(d.shape).to(d.dtype)
        return delta_clean

    def aggregate(self, client_states: dict, global_state: dict) -> dict:
        """
        Calcule les deltas (local - global), filtre, puis reconstruit
        le nouvel état global agrégé.
        """
        num_clients = len(client_states)
        if num_clients == 0:
            return global_state

        # Calcul des deltas
        deltas = {
            cid: {k: client_states[cid][k].float() - global_state[k].float()
                  for k in global_state}
            for cid in client_states
        }

        # Filtrage de chaque delta
        filtered = {cid: self._project_out(d) for cid, d in deltas.items()}

        # Moyenne des deltas filtrés
        avg_delta = {k: torch.zeros_like(global_state[k], dtype=torch.float32)
                     for k in global_state}
        for cid in filtered:
            for k in avg_delta:
                avg_delta[k] += filtered[cid][k]
        for k in avg_delta:
            avg_delta[k] /= num_clients

        # Application au modèle global
        new_state = {k: global_state[k].float() + avg_delta[k]
                     for k in global_state}
        return new_state

    def decay_pin(self):
        self.rounds_since_unlearn += 1
        if self.rounds_since_unlearn % 3 == 0:
            old = self.pin_strength
            self.pin_strength = max(self.PIN_FLOOR,
                                    self.pin_strength * self.PIN_DECAY)
            print(f"[SERVER-SURGERY] 📉 Décroissance pin_strength : "
                  f"{old:.3f} → {self.pin_strength:.3f}")


# ══════════════════════════════════════════════════════════════════
#  COUCHE 3 — ANCRAGE DU VECTEUR DE CLASSE (CLASS ACTIVATION PIN)
#  Après chaque FedAvg, le vecteur de sortie de 'Generic' est
#  interpolé vers un vecteur de confusion uniforme.
# ══════════════════════════════════════════════════════════════════
def apply_class_activation_pin(state_dict: dict,
                                target_class: int,
                                pin_strength: float,
                                num_classes: int = NUM_CLASSES) -> dict:
    """
    Identifie la dernière couche Linear et écrase partiellement
    son vecteur de poids pour la classe cible vers une valeur uniforme.
    Le biais correspondant est également neutralisé.
    """
    # Recherche de la dernière couche Linear dans le state_dict
    linear_weight_key = None
    linear_bias_key   = None
    for key in state_dict:
        if key.endswith(".weight"):
            w = state_dict[key]
            if len(w.shape) == 2 and w.shape[0] == num_classes:
                linear_weight_key = key
                linear_bias_key   = key.replace(".weight", ".bias")

    if linear_weight_key is None:
        print("[SERVER-PIN] ⚠️ Couche de sortie introuvable — pin ignoré.")
        return state_dict

    new_state = {k: v.clone() for k, v in state_dict.items()}

    with torch.no_grad():
        w = new_state[linear_weight_key].float()
        # Vecteur cible : logits uniformes (confusion maximale)
        uniform_row = torch.full((w.shape[1],), 0.0)
        w[target_class] = ((1.0 - pin_strength) * w[target_class]
                           + pin_strength * uniform_row)
        new_state[linear_weight_key] = w

        if linear_bias_key in new_state:
            b = new_state[linear_bias_key].float()
            b[target_class] = b[target_class] * (1.0 - pin_strength)
            new_state[linear_bias_key] = b

    print(f"[SERVER-PIN] 📌 Vecteur 'Generic' (classe {target_class}) ancré "
          f"à {pin_strength*100:.1f} %.")
    return new_state


# ══════════════════════════════════════════════════════════════════
#  SERVEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════
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


class FederatedServer:
    def __init__(self, expected_clients=3, input_size=42):
        self.expected_clients = expected_clients
        self.input_size       = input_size
        self.round            = 0

        self.scaler_updates = {}
        self.ready_clients  = set()
        self.updates        = {}

        self.model = IDS_MLP(input_size=self.input_size, num_classes=NUM_CLASSES)

        # ── État de la protection persistante ──
        self.persistent_aggregator: PersistentFedAvg | None = None
        self.unlearning_done       = False   # Passe à True dès que le signal est reçu

        # ─── CHARGEMENT AUTOMATIQUE DU CHECKPOINT ───
        if os.path.exists(CHECKPOINT_PATH):
            print(f"\n[SERVER] 🔄 Checkpoint '{CHECKPOINT_PATH}' trouvé. Restauration...")
            try:
                checkpoint = torch.load(CHECKPOINT_PATH)
                self.model.load_state_dict(checkpoint["state_dict"])
                self.start_round          = checkpoint["round"]
                self.unlearning_done      = checkpoint.get("unlearning_done", False)
                print(f"[SERVER] 🎉 Reprise au Round {self.start_round + 1}. "
                      f"Protection persistante : {'ACTIVE' if self.unlearning_done else 'inactive'}.")
            except Exception as e:
                print(f"[SERVER] ⚠️ Checkpoint illisible ({e}). Init aléatoire.")
                self.start_round     = 0
                self.unlearning_done = False
        else:
            print("\n[SERVER] 🆕 Aucun checkpoint. Initialisation aléatoire.")
            self.start_round = 0

        # ─── RECHARGEMENT DE LA DIRECTION D'OUBLI (si sauvegardée) ───
        if self.unlearning_done and os.path.exists(UNLEARN_DIR_PATH):
            print(f"[SERVER] 🔁 Direction d'oubli rechargée depuis {UNLEARN_DIR_PATH}.")
            with open(UNLEARN_DIR_PATH, "rb") as f:
                unlearn_dir = pickle.load(f)
            pin_strength = 0.30 if self.start_round > 5 else 0.95
            self.persistent_aggregator = PersistentFedAvg(unlearn_dir, pin_strength)

        # ─── MQTT ───
        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self.on_connect
        self.mqtt.on_message  = self.on_message

    # ──────────────────────────────────────────────────────────────
    def compute_global_scaler(self):
        if os.path.exists(SCALER_PATH) and self.start_round > 0:
            print(f"[SERVER] 💾 Réutilisation du Scaler depuis {SCALER_PATH}")
            with open(SCALER_PATH, "rb") as f:
                data = pickle.load(f)
            return data["mean"], data["std"]

        print("\n[SERVER] 📈 Calcul du Federated Normalization Scaler...")
        total = sum(s["n"] for s in self.scaler_updates.values())
        g_mean = sum(s["mean"] * (s["n"] / total) for s in self.scaler_updates.values())
        g_var  = sum((s["var"] + (s["mean"] - g_mean) ** 2) * (s["n"] / total)
                     for s in self.scaler_updates.values())
        g_std  = np.sqrt(g_var + 1e-5)

        with open(SCALER_PATH, "wb") as f:
            pickle.dump({"mean": g_mean, "std": g_std}, f)
        print(f"[SERVER] ✅ Scaler calculé sur {total} échantillons.")
        return g_mean, g_std

    # ──────────────────────────────────────────────────────────────
    def aggregate(self):
        clients = list(self.updates.keys())
        if not clients:
            return

        global_state = self.model.state_dict()

        # ── Vérification : un client a-t-il envoyé le signal d'oubli ? ──
        for cid in clients:
            if self.updates[cid].get("unlearn_direction") is not None:
                unlearn_dir = self.updates[cid]["unlearn_direction"]
                print(f"\n[SERVER] 🎯 Signal d'oubli reçu du Client {cid} "
                      f"(Round {self.round}). Activation de la protection persistante.")

                # Sauvegarde de la direction pour les redémarrages futurs
                with open(UNLEARN_DIR_PATH, "wb") as f:
                    pickle.dump(unlearn_dir, f)

                self.persistent_aggregator = PersistentFedAvg(unlearn_dir,
                                                               pin_strength_init=0.95)
                self.unlearning_done = True

                # Broadcast du signal d'oubli à tous les clients
                signal_payload = pickle.dumps(unlearn_dir)
                self.mqtt.publish("server/unlearn_signal", signal_payload, retain=True)
                print(f"[SERVER] 📡 Direction d'oubli broadcastée à tous les clients.")
                break  # Un seul signal suffit

        # ── Agrégation : filtrée ou standard ──
        if self.unlearning_done and self.persistent_aggregator is not None:
            print(f"[SERVER] 🛡️  Agrégation FILTRÉE (Gradient Surgery actif).")
            client_states = {cid: self.updates[cid]["state_dict"] for cid in clients}
            new_state = self.persistent_aggregator.aggregate(client_states, global_state)

            # Couche 3 : ancrage du vecteur de classe 'Generic'
            new_state = apply_class_activation_pin(
                new_state,
                target_class  = GENERIC_INDEX,
                pin_strength  = self.persistent_aggregator.pin_strength,
                num_classes   = NUM_CLASSES
            )
            self.model.load_state_dict(new_state)

            # Décroissance progressive du pin
            self.persistent_aggregator.decay_pin()

        else:
            # FedAvg standard (avant désapprentissage)
            fedavg_state = {k: torch.zeros_like(v, dtype=torch.float32)
                            for k, v in global_state.items()}
            for cid in clients:
                for k in fedavg_state:
                    fedavg_state[k] += self.updates[cid]["state_dict"][k].float()
            for k in fedavg_state:
                fedavg_state[k] /= len(clients)
            self.model.load_state_dict(fedavg_state)

        # ── Logs métriques ──
        print(f"\n{'='*70}")
        print(f"[SERVER] Agrégation FedAvg finalisée — Round {self.round}/{TOTAL_ROUNDS}")
        print(f"{'='*70}")
        print("\n[SERVER] Métriques Globales Reçues :")
        for cid in sorted(clients):
            u = self.updates[cid]
            print(f"  -> Client {cid:<2} | F1 Macro: {u['f1_macro']:.4f} | "
                  f"Accuracy: {u['accuracy']:.4f}")
            if "classify_metrics" in u:
                print(f"     └─ Détail par classe :")
                for name, m in sorted(u["classify_metrics"].items()):
                    flag = " ⚠️ CIBLE" if name.lower() == "generic" else ""
                    print(f"        • {name:<15} -> F1: {m['f1']:.4f} | "
                          f"P: {m['precision']:.4f} | R: {m['recall']:.4f}{flag}")
        print()

        # ── Sauvegarde checkpoint ──
        torch.save({
            "round"          : self.round,
            "state_dict"     : self.model.state_dict(),
            "unlearning_done": self.unlearning_done,
        }, CHECKPOINT_PATH)
        print(f"[SERVER] 💾 Checkpoint sauvegardé (Round {self.round}).")

    # ──────────────────────────────────────────────────────────────
    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("[SERVER] Connecté au Broker MQTT.")
        for i in range(self.expected_clients):
            client.publish(f"client/scaler_stats/{i}", b"", retain=True)
            client.publish(f"client/ready/{i}",        b"", retain=True)
        client.publish("server/global_scaler",   b"", retain=True)
        client.publish("server/unlearn_signal",  b"", retain=True)

        client.subscribe("client/scaler_stats/+")
        client.subscribe("client/ready/+")
        client.subscribe("client/update/+")

    def on_message(self, client, userdata, msg):
        if len(msg.payload) == 0:
            return
        topic = msg.topic

        if "scaler_stats" in topic:
            cid = topic.split("/")[-1]
            if cid not in self.scaler_updates:
                self.scaler_updates[cid] = pickle.loads(msg.payload)
                print(f"[SERVER] 📊 Stats Scaler reçues du Client {cid} "
                      f"({len(self.scaler_updates)}/{self.expected_clients})")

        elif "ready" in topic:
            cid = topic.split("/")[-1]
            self.ready_clients.add(cid)
            print(f"[SERVER] ⚡ Client {cid} prêt "
                  f"({len(self.ready_clients)}/{self.expected_clients})")

        elif "update" in topic:
            cid = topic.split("/")[-1]
            if cid not in self.updates:
                self.updates[cid] = pickle.loads(msg.payload)
                print(f"[SERVER] ✨ Update reçue du Client {cid} "
                      f"({len(self.updates)}/{self.expected_clients})")

    # ──────────────────────────────────────────────────────────────
    def start(self):
        self.mqtt.connect(MQTT_BROKER, MQTT_PORT)
        self.mqtt.loop_start()

        try:
            print("[SERVER] En attente des stats Scaler des clients...")
            while len(self.scaler_updates) < self.expected_clients:
                time.sleep(0.5)

            g_mean, g_std = self.compute_global_scaler()
            self.mqtt.publish("server/global_scaler",
                              pickle.dumps({"mean": g_mean, "std": g_std}),
                              retain=False)
            print("[SERVER] 🚀 Scaler Fédéré publié. Attente de la sync clients...")

            while len(self.ready_clients) < self.expected_clients:
                time.sleep(0.5)

            # Re-broadcast du signal d'oubli si reprise après désapprentissage
            if self.unlearning_done and os.path.exists(UNLEARN_DIR_PATH):
                with open(UNLEARN_DIR_PATH, "rb") as f:
                    unlearn_dir = pickle.load(f)
                self.mqtt.publish("server/unlearn_signal",
                                  pickle.dumps(unlearn_dir), retain=True)
                print("[SERVER] 📡 Signal d'oubli re-broadcasté (reprise de session).")

            print(f"\n[SERVER] 🎉 Tous les clients synchronisés. "
                  f"Départ depuis le Round {self.start_round + 1}.")
            time.sleep(1)

            start_idx = self.start_round + 1
            if start_idx > TOTAL_ROUNDS:
                print(f"[SERVER] 🛑 Entraînement déjà complet ({TOTAL_ROUNDS} rounds).")
                self.mqtt.loop_stop()
                return

            for r in range(start_idx, TOTAL_ROUNDS + 1):
                self.updates.clear()
                self.round = r
                print(f"\n📢 [SERVER] ─── DEBUT DU ROUND {self.round}/{TOTAL_ROUNDS} ───")

                self.mqtt.publish("server/global", pickle.dumps(self.model.state_dict()))

                while len(self.updates) < self.expected_clients:
                    time.sleep(0.2)

                self.aggregate()
                time.sleep(0.5)

            print(f"\n[SERVER] ✅ Entraînement terminé ({TOTAL_ROUNDS} rounds).")
            self.mqtt.loop_stop()

        except KeyboardInterrupt:
            print("\n[SERVER] Arrêt demandé.")
            self.mqtt.loop_stop()


if __name__ == "__main__":
    server = FederatedServer(expected_clients=3, input_size=42)
    server.start()