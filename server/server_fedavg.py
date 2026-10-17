import sys
import time
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import paho.mqtt.client as mqtt

MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

NUM_CLASSES = 10
TOTAL_ROUNDS = 400  
CHECKPOINT_PATH = "global_checkpoint.pth"
SCALER_PATH = "federated_scaler.pkl"

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

class FederatedServer:
    def __init__(self, expected_clients=3, input_size=42):
        self.expected_clients = expected_clients
        self.input_size = input_size
        self.round = 0
        
        self.scaler_updates = {}
        self.ready_clients = set() 
        self.updates = {}
        
        self.model = IDS_MLP(input_size=self.input_size, num_classes=NUM_CLASSES)
        
        # ─── CHARGEMENT AUTOMATIQUE DU PRÉCÉDENT RUN (SI EXISTANT) ───
        if os.path.exists(CHECKPOINT_PATH):
            print(f"\n[SERVER] 🔄 Sauvegarde trouvée sous '{CHECKPOINT_PATH}'. Restauration des poids...")
            try:
                checkpoint = torch.load(CHECKPOINT_PATH)
                self.model.load_state_dict(checkpoint["state_dict"])
                self.start_round = checkpoint["round"]
                print(f"[SERVER] 🎉 Reprise réussie ! Prêt à démarrer à partir du Round {self.start_round + 1}.")
            except Exception as e:
                print(f"[SERVER] ⚠️ Impossible de charger le checkpoint ({e}). Initialisation aléatoire.")
                self.start_round = 0
        else:
            print("\n[SERVER] 🆕 Aucune sauvegarde précédente détectée. Initialisation aléatoire des poids.")
            self.start_round = 0
        
        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self.on_connect
        self.mqtt.on_message = self.on_message

    def compute_global_scaler(self):
        # Si un scaler est déjà sauvegardé sur le PC, on réutilise directement les anciennes stats
        if os.path.exists(SCALER_PATH) and self.start_round > 0:
            print(f"[SERVER] 💾 Réutilisation du Scaler Fédéré existant stocké dans : {SCALER_PATH}")
            with open(SCALER_PATH, "rb") as f:
                scaler_data = pickle.load(f)
            return scaler_data["mean"], scaler_data["std"]

        print("\n[SERVER] 📈 Collecte finale et calcul du Federated Normalization Scaler...")
        total_samples = 0
        for stats in self.scaler_updates.values():
            total_samples += stats["n"]

        global_mean = np.zeros_like(list(self.scaler_updates.values())[0]["mean"])
        for stats in self.scaler_updates.values():
            global_mean += stats["mean"] * (stats["n"] / total_samples)

        global_var = np.zeros_like(list(self.scaler_updates.values())[0]["var"])
        for stats in self.scaler_updates.values():
            global_var += (stats["var"] + (stats["mean"] - global_mean) ** 2) * (stats["n"] / total_samples)

        global_std = np.sqrt(global_var + 1e-5)
        
        # Sauvegarde sur le PC du nouveau scaler calculé
        with open(SCALER_PATH, "wb") as f:
            pickle.dump({"mean": global_mean, "std": global_std}, f)
            
        print(f"[SERVER] ✅ Scaler Fédéré calculé et enregistré localement sur {total_samples} paquets réseau.")
        return global_mean, global_std

    def aggregate(self):
        clients = list(self.updates.keys())
        global_state = self.model.state_dict()

        clients_a_agreger = clients
        num_clients_actifs = len(clients_a_agreger)
        if num_clients_actifs == 0:
            return

        fedavg_state = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items()}

        for cid in clients_a_agreger:
            local_state = self.updates[cid]["state_dict"]
            for k in fedavg_state:
                fedavg_state[k] += local_state[k].float()

        for k in fedavg_state:
            fedavg_state[k] = fedavg_state[k] / num_clients_actifs

        self.model.load_state_dict(fedavg_state)
        print(f"\n======================================================================")
        print(f"[SERVER] Agrégation FedAvg finalisée pour le Round {self.round}/{TOTAL_ROUNDS}")
        print(f"======================================================================")
        
        print("\n[SERVER] Métriques Globales Reçues pour ce Round (FedAvg) :")
        for cid in sorted(clients):
            update = self.updates[cid]
            print(f"  -> Client {cid:<2} | F1 Macro: {update['f1_macro']:.4f} | Accuracy: {update['accuracy']:.4f}")
            if "classify_metrics" in update:
                print(f"     └─ Détail par classe :")
                for attack_name, metrics in sorted(update["classify_metrics"].items()):
                    print(f"        • {attack_name:<15} -> F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
        print("\n")

        # ─── SAUVEGARDE SUR PC APRÈS CHAQUE AGRÉGATION ACCUMULÉE ───
        checkpoint_data = {
            "round": self.round,
            "state_dict": self.model.state_dict()
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)
        print(f"[SERVER] 💾 Sauvegarde automatique effectuée : {CHECKPOINT_PATH} (Fini Round {self.round})")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("[SERVER] Connecté au Broker MQTT. Nettoyage et synchronisation...")
        
        for i in range(self.expected_clients):
            client.publish(f"client/scaler_stats/{i}", b"", retain=True)
            client.publish(f"client/ready/{i}", b"", retain=True)
        client.publish("server/global_scaler", b"", retain=True)
        
        client.subscribe("client/scaler_stats/+")
        client.subscribe("client/ready/+")  
        client.subscribe("client/update/+")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload

        if len(payload) == 0:
            return

        if "scaler_stats" in topic:
            cid = topic.split("/")[-1]
            if cid not in self.scaler_updates:
                self.scaler_updates[cid] = pickle.loads(payload)
                print(f"[SERVER] 📊 Stats Scaler reçues du Client {cid} ({len(self.scaler_updates)}/{self.expected_clients})")

        elif "ready" in topic:
            cid = topic.split("/")[-1]
            self.ready_clients.add(cid)
            print(f"[SERVER] ⚡ Client {cid} signale que ses DataLoaders sont prêts ({len(self.ready_clients)}/{self.expected_clients})")

        elif "update" in topic:
            cid = topic.split("/")[-1]
            if cid not in self.updates:
                self.updates[cid] = pickle.loads(payload)
                print(f"[SERVER] ✨ Éléments d'apprentissage reçus du Client {cid} ({len(self.updates)}/{self.expected_clients})")

    def start(self):
        self.mqtt.connect(MQTT_BROKER, MQTT_PORT)
        self.mqtt.loop_start()

        try:
            print("[SERVER] En attente du réveil et des données géométriques des clients...")
            while len(self.scaler_updates) < self.expected_clients:
                time.sleep(0.5)

            g_mean, g_std = self.compute_global_scaler()
            scaler_payload = pickle.dumps({"mean": g_mean, "std": g_std})
            
            print("[SERVER] 🚀 Publication du Scaler Fédéré Unique. En attente de la préparation des clients...")
            self.mqtt.publish("server/global_scaler", scaler_payload, retain=False)

            while len(self.ready_clients) < self.expected_clients:
                time.sleep(0.5)

            print("\n[SERVER] 🎉 Tous les clients réels sont synchronisés. Top départ de la boucle globale.")
            time.sleep(1)

            # Ajustement de la boucle pour démarrer là où on s'était arrêté
            start_idx = self.start_round + 1
            if start_idx > TOTAL_ROUNDS:
                print(f"\n[SERVER] 🛑 L'entraînement de {TOTAL_ROUNDS} rounds est déjà validé sur ton disque.")
                self.mqtt.loop_stop()
                return

            for r in range(start_idx, TOTAL_ROUNDS + 1):
                self.updates.clear()
                self.round = r
                print(f"\n📢 [SERVER] ─── DEBUT DU ROUND FEDERE {self.round}/{TOTAL_ROUNDS} ───")
                
                model_payload = pickle.dumps(self.model.state_dict())
                self.mqtt.publish("server/global", model_payload)

                while len(self.updates) < self.expected_clients:
                    time.sleep(0.2)

                self.aggregate()
                time.sleep(0.5) 

            print(f"\n[SERVER] ✅ Entraînement accompli avec succès sur {TOTAL_ROUNDS} rounds.")
            self.mqtt.loop_stop()

        except KeyboardInterrupt:
            print("\n[SERVER] Arrêt demandé.")
            self.mqtt.loop_stop()

if __name__ == "__main__":
    server = FederatedServer(expected_clients=3, input_size=42)
    server.start()