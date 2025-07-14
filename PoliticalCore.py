# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import logging
from collections import deque
from scipy import stats
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from filterpy.kalman import ExtendedKalmanFilter

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler("political_core_optimized.log"), logging.StreamHandler()])

class PoliticalResonanceLayer(nn.Module):
    def __init__(self, d_model=4096):  # Giảm d_model để tăng tốc
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.phase_shift = nn.Parameter(torch.randn(d_model) * 0.1)  # Giảm biên độ
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.05)
        self.activation = nn.GELU()

    def forward(self, x):
        try:
            with autocast():
                x = self.linear(x)
                x = x + torch.cos(self.phase_shift) * x * 0.5  # Giảm tác động phase_shift
                x = self.activation(x)
                x = self.dropout(x)
                return self.norm(x)
        except Exception as e:
            logging.error(f"Error in PoliticalResonanceLayer: {e}")
            return x

class PoliticalPredictor(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=4096, num_layers=16):  # Giảm num_layers
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim // 2)  # Giảm chiều embedding
        self.lstm = nn.LSTM(hidden_dim // 2, hidden_dim, num_layers=4, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=64, dim_feedforward=16384, batch_first=True),
            num_layers=num_layers)
        self.resonance_layers = nn.ModuleList([PoliticalResonanceLayer(hidden_dim) for _ in range(2)])  # Giảm số layer
        self.fc_stability = nn.Linear(hidden_dim, 1)
        self.fc_trust = nn.Linear(hidden_dim, 1)
        self.fc_tension = nn.Linear(hidden_dim, 1)
        self.fc_cohesion = nn.Linear(hidden_dim, 1)
        self.fc_unrest = nn.Linear(hidden_dim, 1)
        self.fc_substitution = nn.Linear(hidden_dim, 1)
        self.fc_default = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        try:
            with autocast():
                x = self.embedding(x)
                _, (h_lstm, _) = self.lstm(x)
                x = self.transformer(h_lstm)
                for layer in self.resonance_layers:
                    x = layer(x)
                x = self.dropout(x)
                outputs = [torch.sigmoid(fc(x[-1])) for fc in [self.fc_stability, self.fc_trust, self.fc_tension,
                                                              self.fc_cohesion, self.fc_unrest, 
                                                              self.fc_substitution, self.fc_default]]
            return outputs
        except Exception as e:
            logging.error(f"Error in PoliticalPredictor: {e}")
            return tuple(torch.zeros(1).to(x.device) for _ in range(7))

class PoliticalAgent:
    def __init__(self, id: str, nation: str, role: str, influence: float, loyalty: float, adaptability: float):
        self.id = id
        self.nation = nation
        self.role = role
        self.influence = max(0, min(1, influence))
        self.loyalty = max(0, min(1, loyalty))
        self.adaptability = max(0, min(1, adaptability))
        self.confidence = 0.5
        self.propaganda_effect = 0.0
        self.dissent_level = 0.0
        self.cohesion = 0.5
        self.unrest_participation = 0.0
        self.currency_substitution = 0.0
        self.debt_stress = 0.0
        self.history = deque(maxlen=30)  # Giảm maxlen để tiết kiệm bộ nhớ
        self.rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=8)  # Giảm tham số

    def update_state(self, context: Dict[str, float], policy: Optional[Dict], network: nx.DiGraph):
        try:
            stability = context.get("stability", 0.5)
            trust = context.get("trust", 0.5)
            tension = context.get("tension", 0.2)
            unrest = context.get("unrest", 0.0)
            substitution = context.get("substitution", 0.0)
            default = context.get("default", 0.0)
            neighbors = list(network.neighbors(self.id))
            neighbor_conf = np.mean([network.nodes[n]["confidence"] for n in neighbors]) if neighbors else 0.5
            unrest_weight = 1.0 if trust > 0.6 else 1.5  # Trọng số động

            if self.role == "elite":
                self.confidence = 0.55 * stability + 0.3 * self.loyalty + 0.15 * neighbor_conf - 0.15 * tension
                self.dissent_level = max(0, 0.1 - stability * self.adaptability)
                self.unrest_participation = max(0, 0.03 - stability * unrest_weight)
                self.currency_substitution = substitution * (1 - self.loyalty) * 0.25
                self.debt_stress = default * (1 - self.adaptability) * 0.3
            elif self.role == "public":
                self.confidence = 0.4 * trust + 0.25 * self.propaganda_effect + 0.2 * neighbor_conf - 0.25 * tension
                self.dissent_level = max(0, 0.2 - trust * self.adaptability + unrest * 0.15)
                self.unrest_participation = unrest * (1 - trust) * 0.4 * unrest_weight
                self.currency_substitution = substitution * (1 - self.confidence) * 0.5
                self.debt_stress = default * (1 - trust) * 0.25
            elif self.role == "media":
                self.confidence = 0.45 * stability + 0.3 * self.influence + 0.25 * neighbor_conf
                self.propaganda_effect += 0.1 * context.get("propaganda", 0.0)
                self.unrest_participation = max(0, unrest * 0.08)
                self.currency_substitution = substitution * 0.15
                self.debt_stress = default * 0.15
            elif self.role == "opposition":
                self.confidence = 0.25 * (1 - stability) + 0.4 * self.influence - 0.25 * tension
                self.dissent_level = min(1, 0.35 + tension - trust * self.adaptability + unrest * 0.25)
                self.unrest_participation = unrest * (1 - stability) * 0.6 * unrest_weight
                self.currency_substitution = substitution * (1 - self.loyalty) * 0.4
                self.debt_stress = default * (1 - stability) * 0.4

            self.cohesion = 0.45 * self.confidence + 0.3 * stability + 0.2 * self.loyalty - unrest * 0.15
            if policy:
                action = policy.get("action", "maintain")
                param = policy.get("param", 0.0)
                if action == "propaganda":
                    self.propaganda_effect += param
                    self.confidence += param * 0.15
                    self.unrest_participation -= param * 0.08
                elif action == "control":
                    self.dissent_level -= param * 0.25
                    self.confidence -= param * 0.08
                    self.unrest_participation -= param * 0.15
                elif action == "reform":
                    self.cohesion += param * 0.15
                    self.adaptability += param * 0.08
                    self.currency_substitution -= param * 0.08
                elif action == "repression":
                    self.dissent_level += param * 0.15
                    self.confidence -= param * 0.15
                    self.unrest_participation += param * 0.2
                    self.debt_stress += param * 0.08
                elif action == "stabilize_currency":
                    self.currency_substitution -= param * 0.25
                    self.confidence += param * 0.1
                elif action == "debt_restructuring":
                    self.debt_stress -= param * 0.25
                    self.confidence += param * 0.08
                    self.cohesion += param * 0.08

            for attr in ["confidence", "cohesion", "dissent_level", "unrest_participation", 
                         "currency_substitution", "debt_stress"]:
                setattr(self, attr, max(0, min(1, getattr(self, attr))))

            self.history.append({
                "confidence": self.confidence,
                "dissent": self.dissent_level,
                "cohesion": self.cohesion,
                "unrest": self.unrest_participation,
                "substitution": self.currency_substitution,
                "debt_stress": self.debt_stress
            })

            if len(self.history) >= 20:  # Giảm số mẫu để tăng tốc
                X = np.array([[h["confidence"], h["dissent"], h["cohesion"], h["unrest"], 
                               h["substitution"], h["debt_stress"]] for h in self.history])
                y = [1 if h["unrest"] > 0.25 else 0 for h in self.history]
                self.rf_classifier.fit(X[:-1], y[:-1])
        except Exception as e:
            logging.error(f"Error in update_state for {self.id}: {e}")

class SocialUnrestLayer:
    def __init__(self):
        self.threshold = 0.6  # Tăng để giảm nhạy
        self.spread_rate = 0.15  # Giảm để cân bằng

    def update(self, context: Dict[str, float], agents: List[PoliticalAgent], network: nx.DiGraph) -> float:
        try:
            dissent_avg = np.mean([a.dissent_level for a in agents])
            tension = context.get("tension", 0.2)
            trust = context.get("trust", 0.5)
            unrest_level = context.get("unrest", 0.0)
            stability = context.get("stability", 0.5)

            weight = 1.0 if trust > 0.6 else 1.2
            if dissent_avg > self.threshold or tension > 0.7:
                unrest_level += self.spread_rate * (dissent_avg + tension - trust) * weight
                for agent in agents:
                    if random.random() < unrest_level * (1 - agent.confidence * stability):
                        agent.unrest_participation += 0.08
                        agent.confidence -= 0.03
                        agent.dissent_level += 0.08
            else:
                unrest_level = max(0, unrest_level - 0.03)

            return min(1, max(0, unrest_level))
        except Exception as e:
            logging.error(f"Error in SocialUnrestLayer: {e}")
            return context.get("unrest", 0.0)

class CurrencySubstitutionLayer:
    def __init__(self):
        self.substitution_threshold = 0.5  # Tăng để giảm nhạy
        self.conversion_rate = 0.2  # Giảm để cân bằng

    def update(self, context: Dict[str, float], agents: List[PoliticalAgent]) -> float:
        try:
            inflation = context.get("inflation", 0.0)
            trust = context.get("trust", 0.5)
            tension = context.get("tension", 0.2)
            substitution_level = context.get("substitution", 0.0)
            stability = context.get("stability", 0.5)

            weight = 1.0 if stability > 0.7 else 1.3
            if inflation > self.substitution_threshold or trust < 0.5:
                substitution_level += self.conversion_rate * (inflation + tension - trust) * weight
                for agent in agents:
                    if random.random() < substitution_level * (1 - agent.loyalty * stability):
                        agent.currency_substitution += 0.08
                        agent.confidence -= 0.02
            else:
                substitution_level = max(0, substitution_level - 0.02)

            return min(1, max(0, substitution_level))
        except Exception as e:
            logging.error(f"Error in CurrencySubstitutionLayer: {e}")
            return context.get("substitution", 0.0)

class DebtDefaultProbabilityLayer:
    def __init__(self):
        self.default_threshold = 0.4  # Tăng để giảm nhạy
        self.rate_spike = 0.15  # Giảm để cân bằng

    def update(self, context: Dict[str, float], agents: List[PoliticalAgent]) -> float:
        try:
            inflation = context.get("inflation", 0.0)
            stability = context.get("stability", 0.5)
            substitution = context.get("substitution", 0.0)
            default_level = context.get("default", 0.0)

            weight = 1.0 if stability > 0.7 else 1.2
            if inflation > self.default_threshold or substitution > 0.5 and stability < 0.8:
                default_level += self.rate_spike * (inflation + substitution - stability) * weight
                for agent in agents:
                    if random.random() < default_level * (1 - agent.confidence):
                        agent.debt_stress += 0.08
                        agent.cohesion -= 0.03
            else:
                default_level = max(0, default_level - 0.015)

            return min(1, max(0, default_level))
        except Exception as e:
            logging.error(f"Error in DebtDefaultProbabilityLayer: {e}")
            return context.get("default", 0.0)

class EconomicFeedbackLayer:
    def __init__(self):
        self.feedback_threshold = 0.3
        self.impact_rate = 0.1

    def update(self, context: Dict[str, float]) -> Dict[str, float]:
        try:
            inflation = context.get("inflation", 0.0)
            unemployment = context.get("unemployment", 0.1)
            gdp_growth = context.get("gdp_growth", 0.0)
            stability = context.get("stability", 0.5)
            trust = context.get("trust", 0.5)

            feedback = {}
            if inflation > self.feedback_threshold:
                feedback["tension"] = self.impact_rate * inflation
                feedback["trust"] = -self.impact_rate * inflation * (1 - stability)
            if unemployment > 0.15:
                feedback["unrest"] = self.impact_rate * unemployment * (1 - trust)
                feedback["dissent"] = self.impact_rate * unemployment
            if gdp_growth < 0:
                feedback["cohesion"] = -self.impact_rate * abs(gdp_growth)
                feedback["trust"] = -self.impact_rate * abs(gdp_growth) * (1 - stability)

            return {k: min(1, max(-1, v)) for k, v in feedback.items()}
        except Exception as e:
            logging.error(f"Error in EconomicFeedbackLayer: {e}")
            return {}

class PoliticalCore:
    def __init__(self, nation: str, agents: int = 2000000, t: float = 0.0):  # Giảm số agent
        self.nation = nation
        self.t = t
        self.agents = [
            PoliticalAgent(
                f"{nation}_{i}", nation, 
                random.choice(["elite", "public", "media", "opposition"]),
                random.uniform(0.4, 0.9), random.uniform(0.3, 0.95), random.uniform(0.2, 0.8)
            ) for i in range(agents)
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = PoliticalPredictor(input_dim=64).to(self.device)
        self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=0.00003, weight_decay=1e-7)
        self.scaler = GradScaler()
        self.context = {
            "stability": 0.82,
            "trust": 0.60,
            "tension": 0.30,
            "propaganda": 0.55,
            "cohesion": 0.68,
            "dissent": 0.15,
            "unrest": 0.0,
            "substitution": 0.0,
            "default": 0.0,
            "inflation": 0.471,
            "unemployment": 0.1,
            "gdp_growth": 0.032
        }
        self.history = []
        self.policy_memory = deque(maxlen=100)
        self.network = nx.DiGraph()
        self.gpr = GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), alpha=0.005)
        self.ekf = self._initialize_ekf()
        self.q_table = {}
        self.unrest_layer = SocialUnrestLayer()
        self.substitution_layer = CurrencySubstitutionLayer()
        self.default_layer = DebtDefaultProbabilityLayer()
        self.economic_feedback_layer = EconomicFeedbackLayer()
        self._initialize_network()

    def _initialize_ekf(self) -> ExtendedKalmanFilter:
        ekf = ExtendedKalmanFilter(dim_x=7, dim_z=4)
        ekf.x = np.array([0.82, 0.60, 0.30, 0.68, 0.0, 0.0, 0.0])
        ekf.P = np.eye(7) * 0.015
        ekf.Q = np.eye(7) * 0.00015
        ekf.R = np.eye(4) * 0.0015
        return ekf

    def _initialize_network(self):
        for agent in self.agents:
            self.network.add_node(agent.id, **{k: getattr(agent, k) for k in 
                                              ["confidence", "dissent_level", "cohesion", 
                                               "unrest_participation", "currency_substitution", 
                                               "debt_stress", "role"]})
        for i, agent in enumerate(self.agents):
            for j in random.sample(range(len(self.agents)), min(30, len(self.agents))):  # Giảm số kết nối
                if i != j:
                    weight = random.uniform(0.1, 0.4) if agent.role == self.agents[j].role else random.uniform(0.05, 0.25)
                    self.network.add_edge(agent.id, self.agents[j].id, weight=weight)

    def generate_policy(self, context: Dict[str, float]) -> Dict[str, float]:
        try:
            state = [context[k] for k in ["stability", "trust", "tension", "cohesion", "dissent", 
                                          "unrest", "substitution", "default", "inflation"]]
            state_hash = str(round(sum(state) * 1000) / 1000)
            q_values = self.q_table.get(state_hash, {})

            candidates = []
            if context["unrest"] > 0.25 or context["dissent"] > 0.25:
                candidates.extend([
                    {"action": "control", "param": 0.25, "duration": 6, "target": "all", "prob": 0.35},
                    {"action": "propaganda", "param": 0.25, "duration": 8, "target": "public", "prob": 0.25},
                    {"action": "repression", "param": 0.2, "duration": 4, "target": "opposition", "prob": 0.2}
                ])
            elif context["substitution"] > 0.3:
                candidates.extend([
                    {"action": "stabilize_currency", "param": 0.25, "duration": 12, "target": "all", "prob": 0.4},
                    {"action": "propaganda", "param": 0.2, "duration": 8, "target": "public", "prob": 0.3}
                ])
            elif context["default"] > 0.2:
                candidates.extend([
                    {"action": "debt_restructuring", "param": 0.2, "duration": 15, "target": "all", "prob": 0.35},
                    {"action": "reform", "param": 0.15, "duration": 20, "target": "all", "prob": 0.3}
                ])
            elif context["trust"] < 0.5 or context["inflation"] > 0.5:
                candidates.extend([
                    {"action": "propaganda", "param": 0.3, "duration": 10, "target": "public", "prob": 0.4},
                    {"action": "reform", "param": 0.15, "duration": 15, "target": "all", "prob": 0.3}
                ])
            else:
                candidates.append({"action": "maintain", "param": 0.0, "duration": 8, "target": "all", "prob": 1.0})

            if random.random() < 0.08 or not q_values:
                return random.choice(candidates)

            q_values = {c["action"]: q_values.get(c["action"], 0) for c in candidates}
            best_action = max(q_values, key=q_values.get) if q_values else random.choice(candidates)["action"]
            return next(c for c in candidates if c["action"] == best_action)
        except Exception as e:
            logging.error(f"Error in generate_policy: {e}")
            return {"action": "maintain", "param": 0.0, "duration": 8, "target": "all"}

    def apply_policy(self, policy: Dict[str, float]):
        try:
            for agent in random.sample(self.agents, len(self.agents) // 2):  # Giảm số agent cập nhật
                agent.update_state(self.context, policy, self.network)
            self.policy_memory.append({"policy": policy, "t": self.t, "context": self.context.copy()})
        except Exception as e:
            logging.error(f"Error in apply_policy: {e}")

    def evaluate_policy(self):
        try:
            for policy_entry in list(self.policy_memory):
                t_start = policy_entry["t"]
                if self.t - t_start >= policy_entry["policy"]["duration"]:
                    past_context = policy_entry["context"]
                    now = self.context
                    deltas = {k: now[k] - past_context[k] if k in ["stability", "trust", "cohesion"] else 
                              past_context[k] - now[k] for k in ["stability", "trust", "tension", "cohesion", 
                                                                 "dissent", "unrest", "substitution", "default"]}
                    success_score = sum(0.125 * deltas[k] for k in deltas)

                    policy = policy_entry["policy"]
                    state_hash = str(round(sum([past_context[k] for k in ["stability", "trust", "tension", "unrest"]]) * 1000) / 1000)
                    if state_hash not in self.q_table:
                        self.q_table[state_hash] = {}
                    q = self.q_table[state_hash].get(policy["action"], 0)
                    self.q_table[state_hash][policy["action"]] = q + 0.08 * (success_score + 0.9 * max(self.q_table[state_hash].values(), default=0) - q)
                    self.policy_memory.remove(policy_entry)
        except Exception as e:
            logging.error(f"Error in evaluate_policy: {e}")

    def train_predictor(self):
        try:
            if len(self.history) >= 20:
                X = np.array([[h["t"], *(h[k] for k in ["stability", "trust", "tension", "cohesion", "dissent", 
                                                         "unrest", "substitution", "default", "inflation", 
                                                         "unemployment", "gdp_growth"]),
                              np.mean([a.confidence for a in self.agents]),
                              np.mean([a.influence for a in self.agents if a.role == "elite"]),
                              np.mean([a.loyalty for a in self.agents if a.role == "public"]),
                              np.mean([a.dissent_level for a in self.agents if a.role == "opposition"])] 
                             for h in self.history[-20:]])
                y = np.array([[h[k] for k in ["stability", "trust", "tension", "cohesion", 
                                              "unrest", "substitution", "default"]] 
                             for h in self.history[-20:]])
                X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
                y_torch = torch.tensor(y, dtype=torch.float32).to(self.device)

                with autocast():
                    outputs = self.predictor(X_torch.unsqueeze(0))
                    loss = sum(nn.MSELoss()(pred.squeeze(), y_torch[:, i]) for i, pred in enumerate(outputs))
                    loss += 0.00005 * sum(p.pow(2).sum() for p in self.predictor.parameters())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        except Exception as e:
            logging.error(f"Error in train_predictor: {e}")

    def simulate_step(self, t: float) -> Dict[str, float]:
        try:
            policy = self.generate_policy(self.context)
            self.apply_policy(policy)

            feedback = self.economic_feedback_layer.update(self.context)
            for k, v in feedback.items():
                self.context[k] = min(1, max(0, self.context.get(k, 0) + v))

            self.context["unrest"] = self.unrest_layer.update(self.context, self.agents, self.network)
            self.context["substitution"] = self.substitution_layer.update(self.context, self.agents)
            self.context["default"] = self.default_layer.update(self.context, self.agents)

            for agent in self.agents:
                self.network.nodes[agent.id].update({
                    k: getattr(agent, k) for k in ["confidence", "dissent_level", "cohesion", 
                                                   "unrest_participation", "currency_substitution", "debt_stress"]
                })

            pred_input = np.array([[t, *(self.context.get(k, 0) for k in ["stability", "trust", "tension", 
                                                                          "cohesion", "dissent", "unrest", 
                                                                          "substitution", "default", "inflation", 
                                                                          "unemployment", "gdp_growth"]),
                                   np.mean([a.confidence for a in self.agents]),
                                   np.mean([a.influence for a in self.agents if a.role == "elite"]),
                                   np.mean([a.loyalty for a in self.agents if a.role == "public"]),
                                   np.mean([a.dissent_level for a in self.agents if a.role == "opposition"])] * 64])
            pred_torch = torch.tensor(pred_input, dtype=torch.float32).to(self.device)
            stability, trust, tension, cohesion, unrest, substitution, default = self.predictor(pred_torch.unsqueeze(0))

            self.context.update({
                "stability": stability.item(),
                "trust": trust.item(),
                "tension": tension.item(),
                "cohesion": cohesion.item(),
                "dissent": np.mean([a.dissent_level for a in self.agents]),
                "unrest": unrest.item(),
                "substitution": substitution.item(),
                "default": default.item()
            })

            z = np.array([self.context["stability"], self.context["trust"], self.context["tension"], self.context["cohesion"]])
            self.ekf.update(z)
            self.ekf.predict()
            self.context["stability"], self.context["trust"], self.context["tension"], self.context["cohesion"], \
            self.context["unrest"], self.context["substitution"], self.context["default"] = self.ekf.x

            if len(self.history) >= 20:
                X_gpr = np.array([[h[k] for k in ["stability", "trust", "tension", "cohesion", 
                                                  "unrest", "substitution", "default"]] 
                                  for h in self.history[-20:]])
                y_gpr = np.array([1 if h["policy"]["action"] in ["propaganda", "reform", "stabilize_currency"] else 0 
                                  for h in self.history[-20:]])
                self.gpr.fit(X_gpr, y_gpr)

            self.evaluate_policy()
            self.train_predictor()

            result = {
                "t": t,
                **{k: self.context[k] for k in ["stability", "trust", "tension", "cohesion", 
                                                "dissent", "unrest", "substitution", "default"]},
                "policy": policy
            }
            self.history.append(result)
            return result
        except Exception as e:
            logging.error(f"Error in simulate_step: {e}")
            return {
                "t": t,
                "stability": 0.5,
                "trust": 0.5,
                "tension": 0.3,
                "cohesion": 0.5,
                "dissent": 0.1,
                "unrest": 0.0,
                "substitution": 0.0,
                "default": 0.0
            }
