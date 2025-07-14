# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import cupy as cp
import networkx as nx
import torch
import torch.nn as nn
import logging
from collections import defaultdict
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("shadow_economy.log"), logging.StreamHandler()])

class ShadowAgent:
    def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
        self.id = id
        self.nation = nation
        self.wealth = max(0, wealth)  # Tài sản ngầm (tiền mặt, vàng, tài sản không khai báo)
        self.trust_government = max(0, min(1, trust_government))  # Niềm tin vào chính phủ
        self.gold_holdings = 0.0  # Vàng tích trữ
        self.cash_holdings = wealth  # Tiền mặt
        self.black_market_flow = 0.0  # Dòng giao dịch chợ đen
        self.trust_links = {}  # Dictionary lưu mức tin tưởng với các ShadowAgent khác
        self.stress_index = 0.0  # Chỉ số căng thẳng (tăng khi bị điều tra)
        self.activity_log = []  # Lịch sử giao dịch ngầm

    def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
        """Cập nhật niềm tin vào chính phủ dựa trên lạm phát, ổn định chính trị và scandal."""
        try:
            # Niềm tin giảm nếu lạm phát cao hoặc có scandal
            trust_delta = -0.2 * max(0, inflation - 0.05) - 0.3 * scandal_factor
            trust_delta += 0.1 * government_stability
            self.trust_government = max(0, min(1, self.trust_government + trust_delta))
            logging.debug(f"ShadowAgent {self.id}: Trust updated to {self.trust_government:.3f}")
        except Exception as e:
            logging.error(f"Error in update_trust for {self.id}: {e}")
            self.trust_government = 0.5

    def move_wealth_to_gold(self, gold_price: float):
        """Chuyển tài sản sang vàng nếu mất niềm tin hoặc lạm phát cao."""
        try:
            if self.trust_government < 0.3 or random.random() < 0.1:  # Xác suất ngẫu nhiên để mô phỏng tâm lý
                gold_amount = min(self.cash_holdings * 0.5, self.wealth * 0.3) / gold_price
                self.gold_holdings += gold_amount
                self.cash_holdings -= gold_amount * gold_price
                self.wealth = self.cash_holdings + self.gold_holdings * gold_price
                self.activity_log.append({"action": "buy_gold", "amount": gold_amount, "price": gold_price})
                logging.info(f"ShadowAgent {self.id}: Moved {gold_amount:.2f} to gold at {gold_price:.2f}")
        except Exception as e:
            logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    def increase_black_market_trade(self, market_demand: float):
        """Tăng giao dịch chợ đen dựa trên nhu cầu thị trường và niềm tin thấp."""
        try:
            if self.trust_government < 0.4:
                trade_volume = self.wealth * min(0.2, market_demand * (1 - self.trust_government))
                self.black_market_flow += trade_volume
                self.cash_holdings += trade_volume * 0.8  # Lợi nhuận từ chợ đen (80% hiệu quả)
                self.wealth = self.cash_holdings + self.gold_holdings
                self.activity_log.append({"action": "black_market_trade", "volume": trade_volume})
                logging.info(f"ShadowAgent {self.id}: Black market trade volume {trade_volume:.2f}")
        except Exception as e:
            logging.error(f"Error in increase_black_market_trade for {self.id}: {e}")

    def face_investigation(self, detection_prob: float):
        """Đối mặt với điều tra từ chính phủ, có thể mất tài sản hoặc tăng stress."""
        try:
            if random.random() < detection_prob:
                penalty = self.wealth * 0.3  # Phạt 30% tài sản nếu bị phát hiện
                self.wealth -= penalty
                self.cash_holdings = max(0, self.cash_holdings - penalty)
                self.stress_index = min(1, self.stress_index + 0.4)
                self.activity_log.append({"action": "investigated", "penalty": penalty})
                logging.warning(f"ShadowAgent {self.id}: Investigated, lost {penalty:.2f}, stress {self.stress_index:.3f}")
                return penalty
            return 0.0
        except Exception as e:
            logging.error(f"Error in face_investigation for {self.id}: {e}")
            return 0.0

    def interact(self, other_agents: List['ShadowAgent'], shadow_liquidity_pool: float):
        """Tương tác với các ShadowAgent khác qua mạng niềm tin."""
        try:
            for other in other_agents:
                if other.id not in self.trust_links:
                    self.trust_links[other.id] = random.uniform(0.3, 0.7)  # Khởi tạo niềm tin ngẫu nhiên
                if self.trust_links[other.id] > 0.5 and random.random() < 0.2:
                    trade_amount = min(self.cash_holdings, other.cash_holdings) * 0.1
                    self.cash_holdings -= trade_amount
                    other.cash_holdings += trade_amount * 0.95  # 5% phí giao dịch ngầm
                    self.black_market_flow += trade_amount
                    other.black_market_flow += trade_amount
                    self.activity_log.append({"action": "trade", "with": other.id, "amount": trade_amount})
                    logging.debug(f"ShadowAgent {self.id}: Traded {trade_amount:.2f} with {other.id}")
        except Exception as e:
            logging.error(f"Error in interact for {self.id}: {e}")

class ShadowEconomy:
    def __init__(self, nation: str, shadow_agent_count: int = 100000):
        self.nation = nation
        self.agents = [ShadowAgent(f"shadow_{nation}_{i}", nation, random.uniform(1e3, 1e6)) 
                       for i in range(shadow_agent_count)]
        self.shadow_trust_graph = nx.DiGraph()  # Mạng niềm tin
        self.shadow_liquidity_pool = 0.0  # Hồ thanh khoản ngầm
        self.cpi_impact = 0.0  # Ảnh hưởng đến CPI
        self.tax_loss = 0.0  # Thất thu thuế
        self.gold_price = 1800.0  # Giá vàng mặc định
        self.build_trust_graph()

    def build_trust_graph(self):
        """Xây dựng mạng niềm tin giữa các ShadowAgent."""
        try:
            for agent in self.agents:
                self.shadow_trust_graph.add_node(agent.id, trust=agent.trust_government)
                connections = random.sample(self.agents, min(5, len(self.agents)-1))
                for other in connections:
                    if other.id != agent.id:
                        trust_level = agent.trust_links.get(other.id, random.uniform(0.3, 0.7))
                        self.shadow_trust_graph.add_edge(agent.id, other.id, weight=trust_level)
            logging.info(f"ShadowEconomy {self.nation}: Built trust graph with {len(self.agents)} nodes")
        except Exception as e:
            logging.error(f"Error in build_trust_graph for {self.nation}: {e}")

    def update(self, inflation: float, government_stability: float, scandal_factor: float, 
               market_demand: float, detection_prob: float = 0.01):
        """Cập nhật trạng thái nền kinh tế ngầm."""
        try:
            total_black_market_flow = 0.0
            total_tax_loss = 0.0
            self.shadow_liquidity_pool = sum(agent.cash_holdings for agent in self.agents)

            for agent in self.agents:
                # Cập nhật niềm tin
                agent.update_trust(inflation, government_stability, scandal_factor)
                
                # Chuyển tài sản sang vàng nếu lạm phát cao hoặc mất niềm tin
                if inflation > 0.05 or agent.trust_government < 0.3:
                    agent.move_wealth_to_gold(self.gold_price)
                
                # Tăng giao dịch chợ đen
                agent.increase_black_market_trade(market_demand)
                
                # Đối mặt điều tra
                penalty = agent.face_investigation(detection_prob)
                total_tax_loss += penalty
                
                total_black_market_flow += agent.black_market_flow

            # Tác động đến CPI (giá chợ đen làm tăng lạm phát cảm nhận)
            self.cpi_impact = total_black_market_flow / (self.shadow_liquidity_pool + 1e-6) * 0.1
            self.tax_loss = total_tax_loss
            
            # Tương tác giữa các agent
            for agent in self.agents:
                agent.interact(self.agents, self.shadow_liquidity_pool)

            logging.info(f"ShadowEconomy {self.nation}: CPI impact {self.cpi_impact:.3f}, Tax loss {self.tax_loss:.2f}")
        except Exception as e:
            logging.error(f"Error in update for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của nền kinh tế ngầm."""
        try:
            return {
                "total_wealth": sum(agent.wealth for agent in self.agents),
                "gold_holdings": sum(agent.gold_holdings for agent in self.agents),
                "black_market_flow": sum(agent.black_market_flow for agent in self.agents),
                "cpi_impact": self.cpi_impact,
                "tax_loss": self.tax_loss,
                "liquidity_pool": self.shadow_liquidity_pool
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

    def export_data(self, filename: str = "shadow_economy_data.csv"):
        """Xuất dữ liệu nền kinh tế ngầm."""
        try:
            data = {
                "Agent_ID": [agent.id for agent in self.agents],
                "Wealth": [agent.wealth for agent in self.agents],
                "Gold_Holdings": [agent.gold_holdings for agent in self.agents],
                "Cash_Holdings": [agent.cash_holdings for agent in self.agents],
                "Black_Market_Flow": [agent.black_market_flow for agent in self.agents],
                "Trust_Government": [agent.trust_government for agent in self.agents],
                "Stress_Index": [agent.stress_index for agent in self.agents]
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"ShadowEconomy {self.nation}: Exported data to {filename}")
        except Exception as e:
            logging.error(f"Error in export_data for {self.nation}: {e}")

class ShadowGAT(nn.Module):
    """Graph Attention Network để mô hình hóa mạng niềm tin trong shadow economy."""
    def __init__(self, in_dim: int = 8, hidden_dim: int = 16, num_heads: int = 4):
        super().__init__()
        self.gat1 = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads)
        self.gat2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, adj_matrix):
        """x: [num_nodes, in_dim], adj_matrix: [num_nodes, num_nodes]"""
        try:
            x = self.fc(x)
            x, _ = self.gat1(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0), key_padding_mask=~adj_matrix.bool())
            x = self.dropout(x.squeeze(0))
            x, _ = self.gat2(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0), key_padding_mask=~adj_matrix.bool())
            x = self.norm(x.squeeze(0))
            return x
        except Exception as e:
            logging.error(f"Error in ShadowGAT forward: {e}")
            return x

# Tích hợp ShadowEconomy vào hệ thống chính
def integrate_shadow_economy(core, nation_name: str, shadow_scale: int = 100000):
    """Tích hợp ShadowEconomy vào VoTranhAbyssCoreMicro."""
    try:
        core.shadow_economies = getattr(core, 'shadow_economies', {})
        core.shadow_economies[nation_name] = ShadowEconomy(nation_name, shadow_scale)
        core.shadow_gat = ShadowGAT().to(core.device)
        logging.info(f"Integrated ShadowEconomy for {nation_name} with {shadow_scale} agents")
    except Exception as e:
        logging.error(f"Error in integrate_shadow_economy for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm shadow economy
def enhanced_reflect_economy(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                            R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        # Gọi hàm reflect_economy gốc
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        # Cập nhật shadow economy
        if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
            shadow_economy = self.shadow_economies[nation_name]
            shadow_economy.update(
                inflation=space["inflation"],
                government_stability=space.get("institutions", 0.7),
                scandal_factor=0.1 if random.random() < 0.05 else 0.0,  # 5% xác suất scandal
                market_demand=self.global_context.get("pmi", 0.5),
                detection_prob=0.01 * space.get("institutions", 0.7)  # Tỷ lệ phát hiện tỷ lệ thuận với thể chế
            )
            
            # Tác động của shadow economy lên hệ thống chính
            shadow_metrics = shadow_economy.get_metrics()
            space["inflation"] += shadow_metrics["cpi_impact"]  # Ảnh hưởng đến CPI
            observer["GDP"] *= (1 - shadow_metrics["tax_loss"] / (observer["GDP"] + 1e-6))  # Giảm GDP do thất thu thuế
            self.global_context["tax_rate"] *= (1 - 0.1 * shadow_metrics["tax_loss"] / (observer["GDP"] + 1e-6))  # Giảm hiệu quả thuế
            
            # Cập nhật mạng niềm tin bằng GAT
            trust_features = torch.tensor([[agent.trust_government, agent.wealth, agent.gold_holdings, 
                                          agent.cash_holdings, agent.black_market_flow, agent.stress_index,
                                          space["inflation"], self.global_context["pmi"]]
                                         for agent in shadow_economy.agents], dtype=torch.float32).to(self.device)
            adj_matrix = torch.tensor(nx.to_numpy_array(shadow_economy.shadow_trust_graph), dtype=torch.float32).to(self.device)
            updated_trust = self.shadow_gat(trust_features, adj_matrix)
            for i, agent in enumerate(shadow_economy.agents):
                agent.trust_government = max(0, min(1, updated_trust[i, 0].item()))
            
            # Thêm thông tin shadow economy vào kết quả
            result["Shadow_Economy"] = {
                "total_wealth": shadow_metrics["total_wealth"],
                "gold_holdings": shadow_metrics["gold_holdings"],
                "black_market_flow": shadow_metrics["black_market_flow"],
                "cpi_impact": shadow_metrics["cpi_impact"],
                "tax_loss": shadow_metrics["tax_loss"]
            }
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy)

# Ví dụ sử dụng
if __name__ == "__main__":
    # Giả lập core và nation
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    # Tích hợp shadow economy
    integrate_shadow_economy(core, "Vietnam")
    
    # Mô phỏng một bước
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    # Xuất dữ liệu shadow economy
    if "Vietnam" in core.shadow_economies:
        core.shadow_economies["Vietnam"].export_data("shadow_economy_vietnam.csv")
    
    print(f"Shadow Economy Metrics: {result.get('Shadow_Economy', {})}")
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import cupy as cp
import networkx as nx
import torch
import torch.nn as nn
import logging
from collections import deque
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("cultural_inertia.log"), logging.StreamHandler()])

class CulturalInertia:
    def __init__(self, inertia: float = 0.5, memory_length: int = 30):
        self.cultural_inertia = max(0, min(1, inertia))  # Chỉ số quán tính văn hóa
        self.behavior_memory = deque(maxlen=memory_length)  # Lưu hành vi cũ
        self.inertia_shock = 0.0  # Mức độ sốc khi chính sách thay đổi nhanh
        self.trust_influence = 0.0  # Ảnh hưởng từ mạng niềm tin

    def update_inertia(self, social_influence: float, policy_change_magnitude: float):
        """Cập nhật quán tính dựa trên ảnh hưởng xã hội và thay đổi chính sách."""
        try:
            # Quán tính tăng nếu ảnh hưởng xã hội mạnh, giảm nếu chính sách thay đổi lớn
            inertia_delta = 0.1 * social_influence - 0.2 * policy_change_magnitude
            self.cultural_inertia = max(0, min(1, self.cultural_inertia + inertia_delta))
            # Tính sốc quán tính nếu chính sách thay đổi quá nhanh
            self.inertia_shock = policy_change_magnitude * self.cultural_inertia if policy_change_magnitude > 0.3 else 0.0
            logging.debug(f"CulturalInertia updated: inertia={self.cultural_inertia:.3f}, shock={self.inertia_shock:.3f}")
        except Exception as e:
            logging.error(f"Error in update_inertia: {e}")
            self.cultural_inertia = 0.5

    def adjust_behavior(self, new_behavior: Dict[str, float]) -> Dict[str, float]:
        """Điều chỉnh hành vi mới dựa trên quán tính văn hóa."""
        try:
            if not self.behavior_memory:
                self.behavior_memory.append(new_behavior)
                return new_behavior

            past_behavior = self.behavior_memory[-1]
            adjusted_behavior = {}
            for key in new_behavior:
                adjusted_behavior[key] = (1 - self.cultural_inertia) * new_behavior[key] + self.cultural_inertia * past_behavior.get(key, 0.0)
            
            self.behavior_memory.append(adjusted_behavior)
            if self.inertia_shock > 0.2:
                # Phản ứng tiêu cực do sốc quán tính
                for key in adjusted_behavior:
                    adjusted_behavior[key] *= (1 - self.inertia_shock * 0.5)
                logging.warning(f"Inertia shock applied: behavior reduced by {self.inertia_shock * 0.5:.3f}")
            
            return adjusted_behavior
        except Exception as e:
            logging.error(f"Error in adjust_behavior: {e}")
            return new_behavior

# Cập nhật HyperAgent để hỗ trợ cultural inertia
def enhance_hyper_agent(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, 
                     trade_flow: float, resilience: float):
            super().__init__(id, nation, role, wealth, innovation, trade_flow, resilience)
            self.inertia = CulturalInertia(inertia=random.uniform(0.3, 0.7))  # Khởi tạo quán tính ngẫu nhiên

        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            """Cập nhật tâm lý với quán tính văn hóa."""
            try:
                new_psychology = super().update_psychology.__wrapped__(self, global_context, nation_space, 
                                                                      volatility_history, gdp_history, 
                                                                      sentiment, market_momentum)
                psych_dict = {
                    "fear_index": self.fear_index,
                    "greed_index": self.greed_index,
                    "complacency_index": self.complacency_index,
                    "hope_index": self.hope_index
                }
                adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                self.fear_index = adjusted_psych["fear_index"]
                self.greed_index = adjusted_psych["greed_index"]
                self.complacency_index = adjusted_psych["complacency_index"]
                self.hope_index = adjusted_psych["hope_index"]
                self.inertia.update_inertia(
                    social_influence=nation_space.get("cultural_economic_factor", 0.8),
                    policy_change_magnitude=global_context.get("policy_change", 0.0)
                )
                logging.debug(f"HyperAgent {self.id}: Psychology adjusted with inertia {self.inertia.cultural_inertia:.3f}")
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

        def update_real_income(self, inflation: float, interest_rate: float, tax_rate: float):
            """Cập nhật thu nhập thực tế với quán tính."""
            try:
                super().update_real_income(inflation, interest_rate, tax_rate)
                income_dict = {"real_income": self.real_income}
                adjusted_income = self.inertia.adjust_behavior(income_dict)
                self.real_income = adjusted_income["real_income"]
                logging.debug(f"HyperAgent {self.id}: Real income adjusted to {self.real_income:.2f}")
            except Exception as e:
                logging.error(f"Error in update_real_income for {self.id}: {e}")

        def apply_policy_effects(self, policy: Dict[str, float]):
            """Áp dụng chính sách với quán tính."""
            try:
                policy_effect = {
                    "wealth": self.wealth,
                    "trade_flow": self.trade_flow,
                    "resilience": self.resilience,
                    "innovation": self.innovation
                }
                super().apply_policy_effects(policy)
                new_effect = {
                    "wealth": self.wealth,
                    "trade_flow": self.trade_flow,
                    "resilience": self.resilience,
                    "innovation": self.innovation
                }
                adjusted_effect = self.inertia.adjust_behavior(new_effect)
                self.wealth = adjusted_effect["wealth"]
                self.trade_flow = adjusted_effect["trade_flow"]
                self.resilience = adjusted_effect["resilience"]
                self.innovation = adjusted_effect["innovation"]
                logging.debug(f"HyperAgent {self.id}: Policy effects adjusted with inertia")
            except Exception as e:
                logging.error(f"Error in apply_policy_effects for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ cultural inertia
def enhance_shadow_agent(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
            super().__init__(id, nation, wealth, trust_government)
            self.inertia = CulturalInertia(inertia=random.uniform(0.4, 0.8))  # Quán tính cao hơn trong shadow economy

        def move_wealth_to_gold(self, gold_price: float):
            """Chuyển tài sản sang vàng với quán tính."""
            try:
                new_behavior = {"gold_holdings": min(self.cash_holdings * 0.5, self.wealth * 0.3) / gold_price}
                adjusted_behavior = self.inertia.adjust_behavior(new_behavior)
                gold_amount = adjusted_behavior["gold_holdings"]
                self.gold_holdings += gold_amount
                self.cash_holdings -= gold_amount * gold_price
                self.wealth = self.cash_holdings + self.gold_holdings * gold_price
                self.activity_log.append({"action": "buy_gold", "amount": gold_amount, "price": gold_price})
                logging.info(f"ShadowAgent {self.id}: Moved {gold_amount:.2f} to gold with inertia")
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

        def increase_black_market_trade(self, market_demand: float):
            """Tăng giao dịch chợ đen với quán tính."""
            try:
                new_behavior = {"black_market_flow": self.wealth * min(0.2, market_demand * (1 - self.trust_government))}
                adjusted_behavior = self.inertia.adjust_behavior(new_behavior)
                trade_volume = adjusted_behavior["black_market_flow"]
                self.black_market_flow += trade_volume
                self.cash_holdings += trade_volume * 0.8
                self.wealth = self.cash_holdings + self.gold_holdings
                self.activity_log.append({"action": "black_market_trade", "volume": trade_volume})
                logging.info(f"ShadowAgent {self.id}: Black market trade {trade_volume:.2f} with inertia")
            except Exception as e:
                logging.error(f"Error in increase_black_market_trade for {self.id}: {e}")

        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            """Cập nhật niềm tin với quán tính."""
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                trust_dict = {"trust_government": self.trust_government}
                adjusted_trust = self.inertia.adjust_behavior(trust_dict)
                self.trust_government = adjusted_trust["trust_government"]
                self.inertia.update_inertia(
                    social_influence=0.7,  # Ảnh hưởng xã hội cao trong shadow economy
                    policy_change_magnitude=0.0  # ShadowAgent ít bị ảnh hưởng bởi chính sách chính thức
                )
                logging.debug(f"ShadowAgent {self.id}: Trust adjusted to {self.trust_government:.3f}")
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp CulturalInertia vào VoTranhAbyssCoreMicro
def integrate_cultural_inertia(core, nation_name: str):
    """Tích hợp CulturalInertia vào hệ thống chính."""
    try:
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent(core.HyperAgent)
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent(core.shadow_economies[nation_name].ShadowAgent)
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
                agent.inertia = CulturalInertia(inertia=random.uniform(0.4, 0.8))
        logging.info(f"Integrated CulturalInertia for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_cultural_inertia for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm cultural inertia
def enhanced_reflect_economy_with_inertia(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                         R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        # Gọi hàm reflect_economy hiện tại
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        # Tính chỉ số quán tính trung bình
        hyper_agents = [a for a in self.agents if a.nation == nation_name]
        avg_inertia = np.mean([a.inertia.cultural_inertia for a in hyper_agents]) if hyper_agents else 0.5
        
        # Tác động của quán tính lên kinh tế
        if avg_inertia > 0.7:
            space["consumption"] *= (1 - avg_inertia * 0.3)  # Giảm tiêu dùng do quán tính
            space["resilience"] += avg_inertia * 0.1  # Quán tính tăng khả năng phục hồi ngắn hạn
            result["Insight"]["Psychology"] += f" | High cultural inertia ({avg_inertia:.3f}) slowing economic response."
        
        # Tác động lên shadow economy nếu có
        if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
            shadow_economy = self.shadow_economies[nation_name]
            shadow_inertia = np.mean([a.inertia.cultural_inertia for a in shadow_economy.agents])
            shadow_economy.cpi_impact += shadow_inertia * 0.05  # Quán tính ngầm làm tăng CPI
            result["Shadow_Economy"]["inertia"] = shadow_inertia
        
        # Lưu chỉ số quán tính vào lịch sử
        self.history[nation_name][-1]["cultural_inertia"] = avg_inertia
        result["Cultural_Inertia"] = avg_inertia
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_inertia for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_inertia vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_inertia)

# Xuất dữ liệu quán tính
def export_inertia_data(core, nation_name: str, filename: str = "cultural_inertia_data.csv"):
    """Xuất dữ liệu quán tính văn hóa."""
    try:
        hyper_agents = [a for a in core.agents if a.nation == nation_name]
        data = {
            "Agent_ID": [a.id for a in hyper_agents],
            "Cultural_Inertia": [a.inertia.cultural_inertia for a in hyper_agents],
            "Inertia_Shock": [a.inertia.inertia_shock for a in hyper_agents]
        }
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            shadow_agents = core.shadow_economies[nation_name].agents
            data["Agent_ID"] += [a.id for a in shadow_agents]
            data["Cultural_Inertia"] += [a.inertia.cultural_inertia for a in shadow_agents]
            data["Inertia_Shock"] += [a.inertia.inertia_shock for a in shadow_agents]
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f"CulturalInertia {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_inertia_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    # Giả lập core và nation
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    # Tích hợp shadow economy và cultural inertia
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    
    # Mô phỏng một bước
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    # Xuất dữ liệu
    export_inertia_data(core, "Vietnam", "cultural_inertia_vietnam.csv")
    print(f"Cultural Inertia: {result.get('Cultural_Inertia', 0.0):.3f}")
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("propaganda_layer.log"), logging.StreamHandler()])

class MiniPropagandaNet(nn.Module):
    """Mạng Transformer nhẹ để sinh state narrative."""
    def __init__(self, input_dim: int = 16, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 3)  # Dự đoán [sentiment_boost, trust_boost, narrative_strength]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        try:
            x = self.embedding(x)
            x = self.transformer(x.unsqueeze(0)).squeeze(0)
            x = self.fc(x)
            return self.sigmoid(x)  # [batch, 3]
        except Exception as e:
            logging.error(f"Error in MiniPropagandaNet forward: {e}")
            return torch.zeros(x.size(0), 3).to(x.device)

class PropagandaLayer:
    def __init__(self, nation: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.nation = nation
        self.model = MiniPropagandaNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.device = torch.device(device)
        self.narrative_history = deque(maxlen=30)  # Lưu 30 narrative gần nhất
        self.narrative_collapse_risk = 0.0  # Rủi ro sụp đổ niềm tin

    def generate_narrative(self, context: Dict[str, float]) -> Dict[str, float]:
        """Sinh narrative từ bối cảnh kinh tế."""
        try:
            # Chuẩn bị input: PMI, GDP, inflation, sentiment, v.v.
            input_data = torch.tensor([
                context.get("pmi", 0.5),
                context.get("global_growth", 0.03),
                context.get("global_inflation", 0.02),
                context.get("market_sentiment", 0.0),
                context.get("geopolitical_tension", 0.2),
                context.get("Stock_Volatility", 0.0),
                context.get("market_momentum", 0.0),
                context.get("systemic_risk_score", 0.0),
                context.get("GDP", 1e9) / 1e9,
                context.get("inflation", 0.04),
                context.get("trade", 1.0),
                context.get("resilience", 0.5),
                context.get("fear_index", 0.0),
                context.get("greed_index", 0.0),
                context.get("complacency_index", 0.0),
                context.get("hope_index", 0.0)
            ], dtype=torch.float32).to(self.device).unsqueeze(0)

            with torch.no_grad():
                output = self.model(input_data)[0]
            
            narrative = {
                "title": self._craft_narrative_title(context),
                "sentiment_boost": output[0].item(),
                "trust_boost": output[1].item(),
                "narrative_strength": output[2].item()
            }
            self.narrative_history.append(narrative)
            logging.info(f"PropagandaLayer {self.nation}: Generated narrative: {narrative['title']}, strength={narrative['narrative_strength']:.3f}")
            return narrative
        except Exception as e:
            logging.error(f"Error in generate_narrative for {self.nation}: {e}")
            return {"title": "No narrative", "sentiment_boost": 0.0, "trust_boost": 0.0, "narrative_strength": 0.0}

    def _craft_narrative_title(self, context: Dict[str, float]) -> str:
        """Tạo tiêu đề narrative dựa trên bối cảnh."""
        try:
            if context.get("pmi", 0.5) < 0.4:
                return random.choice([
                    "Economic Challenges Reflect Strategic Restructuring",
                    "Temporary Slowdown Signals Robust Future Growth",
                    "Market Resilience Amid Global Uncertainty"
                ])
            elif context.get("pmi", 0.5) > 0.8:
                return random.choice([
                    "Unprecedented Growth Signals Bright Economic Future",
                    "Booming Markets Reflect Strong Policy Success",
                    "Optimism Drives Record Investment Surge"
                ])
            else:
                return random.choice([
                    "Stable Economy Poised for Steady Growth",
                    "Balanced Policies Foster Market Confidence",
                    "Resilient Markets Navigate Global Shifts"
                ])
        except Exception as e:
            logging.error(f"Error in _craft_narrative_title: {e}")
            return "Stable Economy Amid Uncertainty"

    def train_model(self, context: Dict[str, float], actual_outcome: Dict[str, float]):
        """Huấn luyện mô hình dựa trên hiệu quả narrative."""
        try:
            input_data = torch.tensor([
                context.get("pmi", 0.5),
                context.get("global_growth", 0.03),
                context.get("global_inflation", 0.02),
                context.get("market_sentiment", 0.0),
                context.get("geopolitical_tension", 0.2),
                context.get("Stock_Volatility", 0.0),
                context.get("market_momentum", 0.0),
                context.get("systemic_risk_score", 0.0),
                context.get("GDP", 1e9) / 1e9,
                context.get("inflation", 0.04),
                context.get("trade", 1.0),
                context.get("resilience", 0.5),
                context.get("fear_index", 0.0),
                context.get("greed_index", 0.0),
                context.get("complacency_index", 0.0),
                context.get("hope_index", 0.0)
            ], dtype=torch.float32).to(self.device).unsqueeze(0)

            target = torch.tensor([
                actual_outcome.get("sentiment_boost", 0.0),
                actual_outcome.get("trust_boost", 0.0),
                actual_outcome.get("narrative_strength", 0.0)
            ], dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()
            output = self.model(input_data)[0]
            loss = nn.MSELoss()(output, target)
            loss.backward()
            self.optimizer.step()
            logging.debug(f"PropagandaLayer {self.nation}: Trained with loss {loss.item():.4f}")
        except Exception as e:
            logging.error(f"Error in train_model for {self.nation}: {e}")

    def evaluate_narrative(self, actual_pmi: float, narrative_pmi: float):
        """Đánh giá rủi ro sụp đổ niềm tin nếu narrative sai lệch."""
        try:
            discrepancy = abs(actual_pmi - narrative_pmi)
            self.narrative_collapse_risk = min(1, self.narrative_collapse_risk + 0.2 * discrepancy)
            if discrepancy > 0.3:
                logging.warning(f"PropagandaLayer {self.nation}: High discrepancy {discrepancy:.3f}, collapse risk {self.narrative_collapse_risk:.3f}")
            return self.narrative_collapse_risk
        except Exception as e:
            logging.error(f"Error in evaluate_narrative for {self.nation}: {e}")
            return 0.0

# Cập nhật HyperAgent để hỗ trợ propaganda
def enhance_hyper_agent_for_propaganda(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, 
                     trade_flow: float, resilience: float):
            super().__init__(id, nation, role, wealth, innovation, trade_flow, resilience)
            self.belief_in_narrative = random.uniform(0.3, 0.7)  # Niềm tin vào narrative

        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            """Cập nhật tâm lý với ảnh hưởng từ narrative."""
            try:
                super().update_psychology(global_context, nation_space, volatility_history, gdp_history, 
                                          sentiment, market_momentum)
                narrative = global_context.get("narrative", {"sentiment_boost": 0.0, "trust_boost": 0.0})
                
                # Ảnh hưởng của narrative
                if self.belief_in_narrative > 0.5:
                    self.hope_index += narrative["sentiment_boost"] * self.belief_in_narrative * 0.3
                    self.fear_index -= narrative["sentiment_boost"] * self.belief_in_narrative * 0.2
                    self.policy_response["tax_reduction"] += narrative["trust_boost"] * 0.1
                    self.policy_response["subsidy"] += narrative["trust_boost"] * 0.1
                
                # Quán tính làm chậm thay đổi niềm tin
                if hasattr(self, 'inertia'):
                    belief_dict = {"belief_in_narrative": self.belief_in_narrative}
                    adjusted_belief = self.inertia.adjust_behavior(belief_dict)
                    self.belief_in_narrative = adjusted_belief["belief_in_narrative"]
                
                # Cập nhật niềm tin dựa trên hiệu quả narrative
                collapse_risk = global_context.get("narrative_collapse_risk", 0.0)
                if collapse_risk > 0.7 and random.random() < collapse_risk:
                    self.belief_in_narrative = max(0, self.belief_in_narrative - 0.3)
                    self.fear_index += 0.4
                    self.hope_index -= 0.3
                    logging.warning(f"HyperAgent {self.id}: Narrative collapse, belief reduced to {self.belief_in_narrative:.3f}")
                logging.debug(f"HyperAgent {self.id}: Belief in narrative {self.belief_in_narrative:.3f}")
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ propaganda
def enhance_shadow_agent_for_propaganda(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
            super().__init__(id, nation, wealth, trust_government)
            self.belief_in_narrative = random.uniform(0.1, 0.3)  # ShadowAgent ít tin narrative

        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            """Cập nhật niềm tin với ảnh hưởng từ narrative."""
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                narrative = global_context.get("narrative", {"trust_boost": 0.0})
                if self.belief_in_narrative > 0.2:
                    self.trust_government += narrative["trust_boost"] * self.belief_in_narrative * 0.1
                
                # Quán tính làm chậm thay đổi niềm tin
                if hasattr(self, 'inertia'):
                    belief_dict = {"belief_in_narrative": self.belief_in_narrative}
                    adjusted_belief = self.inertia.adjust_behavior(belief_dict)
                    self.belief_in_narrative = adjusted_belief["belief_in_narrative"]
                
                logging.debug(f"ShadowAgent {self.id}: Belief in narrative {self.belief_in_narrative:.3f}")
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp PropagandaLayer vào VoTranhAbyssCoreMicro
def integrate_propaganda_layer(core, nation_name: str):
    """Tích hợp PropagandaLayer vào hệ thống chính."""
    try:
        core.propaganda_layers = getattr(core, 'propaganda_layers', {})
        core.propaganda_layers[nation_name] = PropagandaLayer(nation_name, core.device)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_propaganda(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
            agent.belief_in_narrative = random.uniform(0.3, 0.7)
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_propaganda(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
                agent.belief_in_narrative = random.uniform(0.1, 0.3)
        
        logging.info(f"Integrated PropagandaLayer for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_propaganda_layer for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm propaganda
def enhanced_reflect_economy_with_propaganda(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                            R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        # Gọi hàm reflect_economy hiện tại
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        # Tạo narrative nếu có PropagandaLayer
        if hasattr(self, 'propaganda_layers') and nation_name in self.propaganda_layers:
            propaganda = self.propaganda_layers[nation_name]
            context = {
                **self.global_context,
                **space,
                "GDP": observer.get("GDP", 1e9),
                "market_momentum": result.get("Insight", {}).get("market_momentum", 0.0),
                "Stock_Volatility": result.get("Volatility", 0.0),
                "systemic_risk_score": result.get("Insight", {}).get("Systemic_Risk_Score", 0.0)
            }
            narrative = propaganda.generate_narrative(context)
            self.global_context["narrative"] = narrative
            
            # Đánh giá narrative
            actual_pmi = self.global_context.get("pmi", 0.5)
            narrative_pmi = actual_pmi * (1 + narrative["sentiment_boost"])
            collapse_risk = propaganda.evaluate_narrative(actual_pmi, narrative_pmi)
            self.global_context["narrative_collapse_risk"] = collapse_risk
            
            # Huấn luyện mô hình propaganda
            actual_outcome = {
                "sentiment_boost": min(0.2, max(0, space.get("market_sentiment", 0.0) - space.get("sentiment_prev", 0.0))),
                "trust_boost": min(0.2, max(0, np.mean([a.policy_response.get("tax_reduction", 0.0) 
                                                        for a in self.agents if a.nation == nation_name]))),
                "narrative_strength": narrative["narrative_strength"] * (1 - collapse_risk)
            }
            propaganda.train_model(context, actual_outcome)
            
            # Tác động của narrative lên hệ thống
            avg_belief = np.mean([a.belief_in_narrative for a in self.agents if a.nation == nation_name])
            space["market_sentiment"] += narrative["sentiment_boost"] * avg_belief * 0.5
            if collapse_risk > 0.7:
                space["market_sentiment"] -= 0.3
                space["fear_index"] += 0.4
                result["Insight"]["Psychology"] += f" | Narrative collapse risk high ({collapse_risk:.3f})."
            
            # Tác động lên shadow economy nếu có
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                shadow_belief = np.mean([a.belief_in_narrative for a in self.shadow_economies[nation_name].agents])
                shadow_economy = self.shadow_economies[nation_name]
                shadow_economy.cpi_impact += narrative["sentiment_boost"] * shadow_belief * 0.1  # Narrative ít ảnh hưởng shadow
                
            # Lưu thông tin propaganda vào kết quả
            result["Propaganda"] = {
                "narrative_title": narrative["title"],
                "sentiment_boost": narrative["sentiment_boost"],
                "trust_boost": narrative["trust_boost"],
                "collapse_risk": collapse_risk,
                "avg_belief": avg_belief
            }
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_propaganda for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_propaganda vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_propaganda)

# Xuất dữ liệu propaganda
def export_propaganda_data(core, nation_name: str, filename: str = "propaganda_data.csv"):
    """Xuất dữ liệu propaganda và niềm tin."""
    try:
        hyper_agents = [a for a in core.agents if a.nation == nation_name]
        data = {
            "Agent_ID": [a.id for a in hyper_agents],
            "Belief_in_Narrative": [a.belief_in_narrative for a in hyper_agents]
        }
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            shadow_agents = core.shadow_economies[nation_name].agents
            data["Agent_ID"] += [a.id for a in shadow_agents]
            data["Belief_in_Narrative"] += [a.belief_in_narrative for a in shadow_agents]
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f"PropagandaLayer {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_propaganda_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    # Giả lập core và nation
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    # Tích hợp các tầng
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    
    # Mô phỏng một bước
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    # Xuất dữ liệu
    export_propaganda_data(core, "Vietnam", "propaganda_vietnam.csv")
    print(f"Propaganda Metrics: {result.get('Propaganda', {})}")
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
import copy
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from multiprocessing import Pool
from collections import defaultdict

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("policy_multiverse.log"), logging.StreamHandler()])

class PolicyMultiverseSimulator:
    def __init__(self, nation: str, num_universes: int = 5):
        self.nation = nation
        self.num_universes = num_universes
        self.universes = []  # Lưu trạng thái của các dòng thời gian
        self.results = defaultdict(list)  # Lưu kết quả mỗi vũ trụ

    def fork_universe(self, core, policy: Dict[str, float], agents: List, context: Dict[str, float]):
        """Tạo các dòng thời gian song song với phản ứng khác nhau."""
        try:
            self.universes = []
            reaction_profiles = ["supportive", "neutral", "resistant", "mixed", "chaotic"]
            
            for i in range(self.num_universes):
                universe_core = copy.deepcopy(core)
                universe_agents = copy.deepcopy(agents)
                
                # Gán phản ứng ngẫu nhiên cho tác nhân
                for agent in universe_agents:
                    profile = reaction_profiles[i]
                    agent.reaction_profile = profile
                    if profile == "supportive":
                        agent.belief_in_narrative = min(1, agent.belief_in_narrative + 0.2)
                    elif profile == "resistant":
                        agent.belief_in_narrative = max(0, agent.belief_in_narrative - 0.2)
                        agent.fear_index += 0.2
                    elif profile == "chaotic":
                        agent.fear_index += random.uniform(-0.3, 0.3)
                        agent.greed_index += random.uniform(-0.3, 0.3)
                    elif profile == "mixed":
                        agent.belief_in_narrative += random.uniform(-0.1, 0.1)
                
                self.universes.append({
                    "core": universe_core,
                    "agents": universe_agents,
                    "policy": copy.deepcopy(policy),
                    "context": copy.deepcopy(context),
                    "reaction_profile": profile
                })
                logging.info(f"PolicyMultiverse {self.nation}: Forked universe {i} with profile {profile}")
        except Exception as e:
            logging.error(f"Error in fork_universe for {self.nation}: {e}")

    def simulate_universe(self, universe: Dict):
        """Mô phỏng một bước trong một vũ trụ."""
        try:
            core = universe["core"]
            agents = universe["agents"]
            policy = universe["policy"]
            context = universe["context"]
            profile = universe["reaction_profile"]
            
            # Áp dụng chính sách
            for agent in agents:
                if hasattr(agent, 'apply_policy_effects'):
                    agent.apply_policy_effects(policy)
            
            # Cập nhật core với agents mới
            core.agents = [a for a in core.agents if a.nation != self.nation] + agents
            
            # Chạy mô phỏng
            result = core.reflect_economy(
                t=context.get("t", 1.0),
                observer=core.nations[self.nation]["observer"],
                space=core.nations[self.nation]["space"],
                R_set=[{"growth": 0.03, "cash_flow": 0.5}],
                nation_name=self.nation
            )
            
            # Tính toán reward và entropy
            reward = result.get("Predicted_Value", {}).get("short_term", 0.0) + \
                     result.get("Resilience", 0.0) * 0.5
            entropy = result.get("Entropy", 0.0)
            
            return {
                "profile": profile,
                "reward": reward,
                "entropy": entropy,
                "result": result
            }
        except Exception as e:
            logging.error(f"Error in simulate_universe for {self.nation}: {e}")
            return {"profile": profile, "reward": 0.0, "entropy": 0.0, "result": {}}

    def run_simulation(self, core, policy: Dict[str, float], context: Dict[str, float], agents: List):
        """Chạy mô phỏng đa vũ trụ và trả về kết quả."""
        try:
            self.fork_universe(core, policy, agents, context)
            
            with Pool(processes=self.num_universes) as pool:
                universe_results = pool.map(self.simulate_universe, self.universes)
            
            for res in universe_results:
                self.results[res["profile"]].append(res)
            
            # Tìm best và worst case
            best_case = max(universe_results, key=lambda x: x["reward"], default={"profile": "none", "reward": 0.0})
            worst_case = min(universe_results, key=lambda x: x["reward"], default={"profile": "none", "reward": 0.0})
            
            logging.info(f"PolicyMultiverse {self.nation}: Best case {best_case['profile']} (reward={best_case['reward']:.3f}), "
                        f"Worst case {worst_case['profile']} (reward={worst_case['reward']:.3f})")
            
            return {
                "best_case": best_case,
                "worst_case": worst_case,
                "all_results": universe_results
            }
        except Exception as e:
            logging.error(f"Error in run_simulation for {self.nation}: {e}")
            return {"best_case": {}, "worst_case": {}, "all_results": []}

    def export_data(self, filename: str = "multiverse_data.csv"):
        """Xuất dữ liệu mô phỏng đa vũ trụ."""
        try:
            data = {
                "Universe": [],
                "Profile": [],
                "Reward": [],
                "Entropy": [],
                "Consumption": [],
                "Resilience": []
            }
            for profile, results in self.results.items():
                for res in results:
                    data["Universe"].append(f"{self.nation}_{profile}_{len(data['Universe'])}")
                    data["Profile"].append(profile)
                    data["Reward"].append(res["reward"])
                    data["Entropy"].append(res["entropy"])
                    data["Consumption"].append(res["result"].get("Insight", {}).get("Consumption", {}).get("base", 0.0))
                    data["Resilience"].append(res["result"].get("Resilience", 0.0))
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"PolicyMultiverse {self.nation}: Exported data to {filename}")
        except Exception as e:
            logging.error(f"Error in export_data for {self.nation}: {e}")

# Cập nhật HyperAgent để hỗ trợ multiverse
def enhance_hyper_agent_for_multiverse(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, 
                     trade_flow: float, resilience: float):
            super().__init__(id, nation, role, wealth, innovation, trade_flow, resilience)
            self.reaction_profile = random.choice(["supportive", "neutral", "resistant"])  # Phản ứng mặc định

        def apply_policy_effects(self, policy: Dict[str, float]):
            """Áp dụng chính sách với phản ứng đa dạng."""
            try:
                super().apply_policy_effects(policy)
                if self.reaction_profile == "supportive":
                    self.wealth *= 1.1
                    self.hope_index += 0.2
                elif self.reaction_profile == "resistant":
                    self.wealth *= 0.95
                    self.fear_index += 0.2
                    self.trade_flow *= 0.9
                elif self.reaction_profile == "chaotic":
                    self.wealth *= random.uniform(0.8, 1.2)
                    self.innovation += random.uniform(-0.1, 0.1)
                logging.debug(f"HyperAgent {self.id}: Applied policy with profile {self.reaction_profile}")
            except Exception as e:
                logging.error(f"Error in apply_policy_effects for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ multiverse
def enhance_shadow_agent_for_multiverse(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
            super().__init__(id, nation, wealth, trust_government)
            self.reaction_profile = random.choice(["neutral", "resistant", "chaotic"])  # ShadowAgent ít hỗ trợ

        def apply_policy_effects(self, policy: Dict[str, float]):
            """Áp dụng chính sách với phản ứng đa dạng trong shadow economy."""
            try:
                if self.reaction_profile == "neutral":
                    self.black_market_flow *= 1.05
                elif self.reaction_profile == "resistant":
                    self.gold_holdings += self.cash_holdings * 0.1 / 1800.0
                    self.cash_holdings *= 0.9
                elif self.reaction_profile == "chaotic":
                    self.black_market_flow += random.uniform(-0.2, 0.2) * self.wealth
                self.wealth = self.cash_holdings + self.gold_holdings * 1800.0
                logging.debug(f"ShadowAgent {self.id}: Applied policy with profile {self.reaction_profile}")
            except Exception as e:
                logging.error(f"Error in apply_policy_effects for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp PolicyMultiverseSimulator vào VoTranhAbyssCoreMicro
def integrate_multiverse_simulator(core, nation_name: str):
    """Tích hợp PolicyMultiverseSimulator vào hệ thống chính."""
    try:
        core.multiverse_simulators = getattr(core, 'multiverse_simulators', {})
        core.multiverse_simulators[nation_name] = PolicyMultiverseSimulator(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_multiverse(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
            agent.reaction_profile = random.choice(["supportive", "neutral", "resistant"])
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_multiverse(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
                agent.reaction_profile = random.choice(["neutral", "resistant", "chaotic"])
        
        logging.info(f"Integrated PolicyMultiverseSimulator for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_multiverse_simulator for {nation_name}: {e}")

# Cập nhật apply_policy_impact để sử dụng multiverse
def enhanced_apply_policy_impact(self, nation_name: str, policy: Dict[str, float]):
    """Áp dụng chính sách với mô phỏng đa vũ trụ."""
    try:
        space = self.nations[nation_name]["space"]
        observer = self.nations[nation_name]["observer"]
        
        # Áp dụng chính sách như bình thường
        VoTranhAbyssCoreMicro.apply_policy_impact(self, nation_name, policy)
        
        # Chạy mô phỏng đa vũ trụ nếu có simulator
        if hasattr(self, 'multiverse_simulators') and nation_name in self.multiverse_simulators:
            multiverse = self.multiverse_simulators[nation_name]
            context = {
                "t": self.t,
                **self.global_context,
                **space,
                "GDP": observer.get("GDP", 1e9)
            }
            agents = [a for a in self.agents if a.nation == nation_name]
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                agents += self.shadow_economies[nation_name].agents
            
            multiverse_result = multiverse.run_simulation(self, policy, context, agents)
            
            # Cập nhật kết quả vào lịch sử
            self.history[nation_name][-1]["multiverse"] = {
                "best_case": multiverse_result["best_case"]["profile"],
                "best_reward": multiverse_result["best_case"]["reward"],
                "worst_case": multiverse_result["worst_case"]["profile"],
                "worst_reward": multiverse_result["worst_case"]["reward"]
            }
            
            # Tác động của best case lên niềm tin
            best_reward = multiverse_result["best_case"]["reward"]
            if best_reward > 0.5:
                space["market_sentiment"] += 0.1
                space["hope_index"] += 0.1
            elif multiverse_result["worst_case"]["reward"] < -0.5:
                space["fear_index"] += 0.2
                space["market_sentiment"] -= 0.1
            
            logging.info(f"Multiverse {nation_name}: Best case {multiverse_result['best_case']['profile']} "
                        f"(reward={best_reward:.3f})")
    except Exception as e:
        logging.error(f"Error in enhanced_apply_policy_impact for {nation_name}: {e}")

# Gắn hàm enhanced_apply_policy_impact vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'apply_policy_impact', enhanced_apply_policy_impact)

# Cập nhật reflect_economy để bao gồm multiverse
def enhanced_reflect_economy_with_multiverse(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                            R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'multiverse_simulators') and nation_name in self.multiverse_simulators:
            multiverse = self.multiverse_simulators[nation_name]
            multiverse_result = self.history[nation_name][-1].get("multiverse", {})
            result["Multiverse"] = {
                "best_case": multiverse_result.get("best_case", "none"),
                "best_reward": multiverse_result.get("best_reward", 0.0),
                "worst_case": multiverse_result.get("worst_case", "none"),
                "worst_reward": multiverse_result.get("worst_reward", 0.0)
            }
            result["Insight"]["Psychology"] += f" | Multiverse best: {multiverse_result.get('best_case', 'none')} " \
                                             f"(reward={multiverse_result.get('best_reward', 0.0):.3f})"
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_multiverse for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_multiverse vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_multiverse)

# Xuất dữ liệu multiverse
def export_multiverse_data(core, nation_name: str, filename: str = "multiverse_data.csv"):
    """Xuất dữ liệu mô phỏng đa vũ trụ."""
    try:
        if hasattr(core, 'multiverse_simulators') and nation_name in core.multiverse_simulators:
            core.multiverse_simulators[nation_name].export_data(filename)
    except Exception as e:
        logging.error(f"Error in export_multiverse_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_multiverse_data(core, "Vietnam", "multiverse_vietnam.csv")
    print(f"Multiverse Metrics: {result.get('Multiverse', {})}")
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import logging
import pandas as pd
from scipy.sparse import csr_matrix
from torch_geometric.nn import GCNConv

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("trust_dynamics.log"), logging.StreamHandler()])

class TrustGCN(nn.Module):
    """Graph Convolutional Network để mô phỏng lan truyền niềm tin."""
    def __init__(self, in_dim: int = 10, hidden_dim: int = 32, out_dim: int = 1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        try:
            x = self.conv1(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            return torch.sigmoid(x)
        except Exception as e:
            logging.error(f"Error in TrustGCN forward: {e}")
            return x

class NetworkedTrustDynamics:
    def __init__(self, nation: str, agent_count: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.nation = nation
        self.trust_graph = nx.DiGraph()
        self.device = torch.device(device)
        self.gcn = TrustGCN().to(self.device)
        self.optimizer = torch.optim.Adam(self.gcn.parameters(), lr=0.001)
        self.agent_count = agent_count
        self.trust_matrix = None

    def build_trust_graph(self, agents: List):
        """Xây dựng mạng niềm tin giữa các tác nhân."""
        try:
            self.trust_graph.clear()
            for agent in agents:
                self.trust_graph.add_node(agent.id, trust=agent.trust_government)
                connections = random.sample(agents, min(10, len(agents)-1))
                for other in connections:
                    if other.id != agent.id:
                        trust_level = random.uniform(0.2, 0.8)
                        self.trust_graph.add_edge(agent.id, other.id, weight=trust_level)
            logging.info(f"NetworkedTrustDynamics {self.nation}: Built trust graph with {len(agents)} nodes")
        except Exception as e:
            logging.error(f"Error in build_trust_graph for {self.nation}: {e}")

    def update_trust(self, agents: List, context: Dict[str, float]):
        """Cập nhật niềm tin qua mạng GCN."""
        try:
            # Chuẩn bị dữ liệu cho GCN
            node_features = torch.tensor([
                [
                    agent.trust_government,
                    agent.wealth / (agent.wealth + 1e6),
                    agent.fear_index,
                    agent.greed_index,
                    agent.hope_index,
                    context.get("pmi", 0.5),
                    context.get("inflation", 0.04),
                    context.get("market_sentiment", 0.0),
                    context.get("Stock_Volatility", 0.0),
                    context.get("geopolitical_tension", 0.2)
                ] for agent in agents
            ], dtype=torch.float32).to(self.device)

            # Chuyển đổi trust_graph sang edge_index
            adj = nx.to_scipy_sparse_array(self.trust_graph, weight='weight', format='csr')
            edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long).to(self.device)

            # Chạy GCN
            self.gcn.eval()
            with torch.no_grad():
                trust_scores = self.gcn(node_features, edge_index).squeeze()

            # Cập nhật niềm tin
            for i, agent in enumerate(agents):
                old_trust = agent.trust_government
                new_trust = trust_scores[i].item()
                if hasattr(agent, 'inertia'):
                    trust_dict = {"trust_government": new_trust}
                    adjusted_trust = agent.inertia.adjust_behavior(trust_dict)
                    new_trust = adjusted_trust["trust_government"]
                agent.trust_government = max(0, min(1, new_trust))
                logging.debug(f"Agent {agent.id}: Trust updated from {old_trust:.3f} to {agent.trust_government:.3f}")

            # Huấn luyện GCN
            self.gcn.train()
            self.optimizer.zero_grad()
            pred_trust = self.gcn(node_features, edge_index).squeeze()
            target_trust = torch.tensor([agent.trust_government for agent in agents], 
                                      dtype=torch.float32).to(self.device)
            loss = nn.MSELoss()(pred_trust, target_trust)
            loss.backward()
            self.optimizer.step()
            logging.debug(f"NetworkedTrustDynamics {self.nation}: GCN trained with loss {loss.item():.4f}")

        except Exception as e:
            logging.error(f"Error in update_trust for {self.nation}: {e}")

    def simulate_scandal(self, agents: List, scandal_factor: float = 0.3):
        """Mô phỏng scandal lan truyền qua mạng niềm tin."""
        try:
            affected_agents = random.sample(agents, int(0.1 * len(agents)))  # 10% bị ảnh hưởng trực tiếp
            for agent in affected_agents:
                agent.trust_government = max(0, agent.trust_government - scandal_factor)
                agent.fear_index += scandal_factor * 0.5
                logging.debug(f"Agent {agent.id}: Trust reduced by scandal to {agent.trust_government:.3f}")

            # Lan truyền qua mạng
            self.update_trust(agents, {})
            logging.info(f"NetworkedTrustDynamics {self.nation}: Scandal simulated, factor={scandal_factor:.3f}")
        except Exception as e:
            logging.error(f"Error in simulate_scandal for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của mạng niềm tin."""
        try:
            trust_values = [data['trust'] for _, data in self.trust_graph.nodes(data=True)]
            return {
                "avg_trust": np.mean(trust_values) if trust_values else 0.0,
                "trust_variance": np.var(trust_values) if trust_values else 0.0,
                "graph_density": nx.density(self.trust_graph)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

    def export_data(self, filename: str = "trust_dynamics_data.csv"):
        """Xuất dữ liệu mạng niềm tin."""
        try:
            data = {
                "Agent_ID": [node for node in self.trust_graph.nodes()],
                "Trust": [self.trust_graph.nodes[node]['trust'] for node in self.trust_graph.nodes()]
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"NetworkedTrustDynamics {self.nation}: Exported data to {filename}")
        except Exception as e:
            logging.error(f"Error in export_data for {self.nation}: {e}")

# Cập nhật HyperAgent để hỗ trợ trust dynamics
def enhance_hyper_agent_for_trust(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            """Cập nhật tâm lý với ảnh hưởng từ mạng niềm tin."""
            try:
                super().update_psychology(global_context, nation_space, volatility_history, gdp_history, 
                                          sentiment, market_momentum)
                trust_influence = self.trust_government - nation_space.get("trust_prev", self.trust_government)
                self.fear_index -= trust_influence * 0.2
                self.hope_index += trust_influence * 0.2
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "fear_index": self.fear_index,
                        "hope_index": self.hope_index
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.fear_index = adjusted_psych["fear_index"]
                    self.hope_index = adjusted_psych["hope_index"]
                logging.debug(f"HyperAgent {self.id}: Psychology adjusted with trust {self.trust_government:.3f}")
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ trust dynamics
def enhance_shadow_agent_for_trust(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            """Cập nhật niềm tin với ảnh hưởng từ mạng."""
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                logging.debug(f"ShadowAgent {self.id}: Trust adjusted to {self.trust_government:.3f}")
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp NetworkedTrustDynamics vào VoTranhAbyssCoreMicro
def integrate_trust_dynamics(core, nation_name: str):
    """Tích hợp NetworkedTrustDynamics vào hệ thống chính."""
    try:
        core.trust_dynamics = getattr(core, 'trust_dynamics', {})
        core.trust_dynamics[nation_name] = NetworkedTrustDynamics(nation_name, len(core.agents))
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_trust(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_trust(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
        
        # Xây dựng trust graph
        agents = [a for a in core.agents if a.nation == nation_name]
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            agents += core.shadow_economies[nation_name].agents
        core.trust_dynamics[nation_name].build_trust_graph(agents)
        
        logging.info(f"Integrated NetworkedTrustDynamics for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_trust_dynamics for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm trust dynamics
def enhanced_reflect_economy_with_trust(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                       R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'trust_dynamics') and nation_name in self.trust_dynamics:
            trust_dynamics = self.trust_dynamics[nation_name]
            agents = [a for a in self.agents if a.nation == nation_name]
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                agents += self.shadow_economies[nation_name].agents
            
            # Cập nhật niềm tin
            context = {**self.global_context, **space}
            trust_dynamics.update_trust(agents, context)
            
            # Mô phỏng scandal ngẫu nhiên
            if random.random() < 0.05:  # 5% xác suất scandal
                trust_dynamics.simulate_scandal(agents, scandal_factor=0.3)
                space["market_sentiment"] -= 0.2
                space["fear_index"] += 0.3
            
            # Tác động của niềm tin lên hệ thống
            trust_metrics = trust_dynamics.get_metrics()
            space["market_sentiment"] += trust_metrics["avg_trust"] * 0.1
            space["resilience"] += trust_metrics["avg_trust"] * 0.05
            if trust_metrics["trust_variance"] > 0.1:
                space["fear_index"] += 0.1
                result["Insight"]["Psychology"] += f" | High trust variance ({trust_metrics['trust_variance']:.3f}) causing instability."
            
            result["Trust_Dynamics"] = trust_metrics
            self.history[nation_name][-1]["trust_metrics"] = trust_metrics
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_trust for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_trust vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_trust)

# Xuất dữ liệu trust dynamics
def export_trust_data(core, nation_name: str, filename: str = "trust_dynamics_data.csv"):
    """Xuất dữ liệu mạng niềm tin."""
    try:
        if hasattr(core, 'trust_dynamics') and nation_name in core.trust_dynamics:
            core.trust_dynamics[nation_name].export_data(filename)
    except Exception as e:
        logging.error(f"Error in export_trust_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_trust_data(core, "Vietnam", "trust_dynamics_vietnam.csv")
    print(f"Trust Dynamics Metrics: {result.get('Trust_Dynamics', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("timewarp_gdp.log"), logging.StreamHandler()])

class TimewarpGDP:
    def __init__(self, nation: str, memory_length: int = 90):
        self.nation = nation
        self.past_gdp_expectations = deque(maxlen=memory_length)  # Lưu kỳ vọng GDP
        self.expectation_shock = 0.0  # Mức độ sốc khi GDP lệch kỳ vọng

    def update_expectation(self, actual_gdp: float, sentiment: float, volatility: float):
        """Cập nhật kỳ vọng GDP dựa trên thực tế và tâm lý thị trường."""
        try:
            expected_gdp = actual_gdp * (1 + sentiment * 0.1 - volatility * 0.05)
            self.past_gdp_expectations.append(expected_gdp)
            
            if len(self.past_gdp_expectations) >= 30:
                past_avg = np.mean(list(self.past_gdp_expectations)[:-1])
                self.expectation_shock = abs(actual_gdp - past_avg) / (past_avg + 1e-6)
            else:
                self.expectation_shock = 0.0
            
            logging.debug(f"TimewarpGDP {self.nation}: Expected GDP {expected_gdp:.2e}, Shock {self.expectation_shock:.3f}")
        except Exception as e:
            logging.error(f"Error in update_expectation for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của TimewarpGDP."""
        try:
            return {
                "latest_expectation": self.past_gdp_expectations[-1] if self.past_gdp_expectations else 0.0,
                "expectation_shock": self.expectation_shock
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ TimewarpGDP
def enhance_hyper_agent_for_timewarp(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            """Cập nhật tâm lý với ảnh hưởng từ kỳ vọng GDP."""
            try:
                super().update_psychology(global_context, nation_space, volatility_history, gdp_history, 
                                          sentiment, market_momentum)
                shock = global_context.get("expectation_shock", 0.0)
                if shock > 0.3:
                    self.fear_index += shock * 0.5
                    self.hope_index -= shock * 0.4
                    self.wealth *= (1 - shock * 0.2)  # Giảm chi tiêu
                elif shock < 0.1 and shock > 0:
                    self.hope_index += 0.2
                    self.wealth *= 1.1  # Tăng chi tiêu
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "fear_index": self.fear_index,
                        "hope_index": self.hope_index
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.fear_index = adjusted_psych["fear_index"]
                    self.hope_index = adjusted_psych["hope_index"]
                logging.debug(f"HyperAgent {self.id}: Psychology adjusted with GDP shock {shock:.3f}")
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

        def update_consumption_state(self):
            """Cập nhật trạng thái tiêu dùng với ảnh hưởng từ kỳ vọng GDP."""
            try:
                super().update_consumption_state()
                shock = global_context.get("expectation_shock", 0.0)
                if shock > 0.3:
                    self.consumption_state = "low" if random.random() < 0.7 else self.consumption_state
                logging.debug(f"HyperAgent {self.id}: Consumption state {self.consumption_state} with shock {shock:.3f}")
            except Exception as e:
                logging.error(f"Error in update_consumption_state for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ TimewarpGDP
def enhance_shadow_agent_for_timewarp(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            """Cập nhật niềm tin với ảnh hưởng từ kỳ vọng GDP."""
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                shock = global_context.get("expectation_shock", 0.0)
                if shock > 0.3:
                    self.trust_government = max(0, self.trust_government - shock * 0.2)
                    self.black_market_flow += self.wealth * shock * 0.1
                logging.debug(f"ShadowAgent {self.id}: Trust adjusted with GDP shock {shock:.3f}")
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

        def move_wealth_to_gold(self, gold_price: float):
            """Chuyển tài sản sang vàng với ảnh hưởng từ kỳ vọng GDP."""
            try:
                super().move_wealth_to_gold(gold_price)
                shock = global_context.get("expectation_shock", 0.0)
                if shock > 0.3:
                    extra_gold = self.cash_holdings * shock * 0.2 / gold_price
                    self.gold_holdings += extra_gold
                    self.cash_holdings -= extra_gold * gold_price
                    self.wealth = self.cash_holdings + self.gold_holdings * gold_price
                    logging.debug(f"ShadowAgent {self.id}: Extra gold {extra_gold:.2f} due to GDP shock {shock:.3f}")
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp TimewarpGDP vào VoTranhAbyssCoreMicro
def integrate_timewarp_gdp(core, nation_name: str):
    """Tích hợp TimewarpGDP vào hệ thống chính."""
    try:
        core.timewarp_gdp = getattr(core, 'timewarp_gdp', {})
        core.timewarp_gdp[nation_name] = TimewarpGDP(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_timewarp(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_timewarp(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
        
        logging.info(f"Integrated TimewarpGDP for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_timewarp_gdp for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm TimewarpGDP
def enhanced_reflect_economy_with_timewarp(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                          R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'timewarp_gdp') and nation_name in self.timewarp_gdp:
            timewarp = self.timewarp_gdp[nation_name]
            timewarp.update_expectation(
                actual_gdp=observer.get("GDP", 1e9),
                sentiment=space.get("market_sentiment", 0.0),
                volatility=result.get("Volatility", 0.0)
            )
            
            # Tác động của kỳ vọng GDP lên hệ thống
            metrics = timewarp.get_metrics()
            self.global_context["expectation_shock"] = metrics["expectation_shock"]
            
            if metrics["expectation_shock"] > 0.3:
                space["consumption"] *= (1 - metrics["expectation_shock"] * 0.3)
                space["market_sentiment"] -= metrics["expectation_shock"] * 0.2
                space["fear_index"] += metrics["expectation_shock"] * 0.4
                result["Insight"]["Psychology"] += f" | GDP expectation shock ({metrics['expectation_shock']:.3f}) causing panic."
            elif metrics["expectation_shock"] < 0.1 and metrics["expectation_shock"] > 0:
                space["consumption"] *= 1.1
                space["hope_index"] += 0.2
            
            # Tác động lên shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_economy.cpi_impact += metrics["expectation_shock"] * 0.1
            
            result["Timewarp_GDP"] = metrics
            self.history[nation_name][-1]["timewarp_metrics"] = metrics
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_timewarp for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_timewarp vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_timewarp)

# Xuất dữ liệu TimewarpGDP
def export_timewarp_data(core, nation_name: str, filename: str = "timewarp_gdp_data.csv"):
    """Xuất dữ liệu kỳ vọng GDP."""
    try:
        if hasattr(core, 'timewarp_gdp') and nation_name in core.timewarp_gdp:
            timewarp = core.timewarp_gdp[nation_name]
            data = {
                "Time": list(range(len(timewarp.past_gdp_expectations))),
                "Expected_GDP": list(timewarp.past_gdp_expectations),
                "Expectation_Shock": [timewarp.expectation_shock] * len(timewarp.past_gdp_expectations)
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"TimewarpGDP {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_timewarp_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_timewarp_data(core, "Vietnam", "timewarp_gdp_vietnam.csv")
    print(f"Timewarp GDP Metrics: {result.get('Timewarp_GDP', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("neocortex_emulator.log"), logging.StreamHandler()])

class NeocortexDeficitEmulator:
    def __init__(self, nation: str, memory_length: int = 30):
        self.nation = nation
        self.stress_history = deque(maxlen=memory_length)  # Lưu lịch sử stress
        self.debt_default_risk = 0.0  # Rủi ro vỡ nợ

    def update_stress(self, agent_wealth: float, income_drop: float, news_negativity: float):
        """Cập nhật mức độ stress dựa trên tài chính và tin tức tiêu cực."""
        try:
            # Mô phỏng theo Kahneman-Tversky: Losses loom larger than gains
            stress = min(1.0, max(0.0, 0.4 * income_drop + 0.3 * news_negativity - 0.2 * (agent_wealth / (agent_wealth + 1e6))))
            self.stress_history.append(stress)
            
            # Tính rủi ro vỡ nợ dựa trên stress cao
            if len(self.stress_history) >= 10:
                avg_stress = np.mean(list(self.stress_history))
                self.debt_default_risk = min(1.0, avg_stress * 1.5 if avg_stress > 0.8 else 0.0)
            else:
                self.debt_default_risk = 0.0
            
            logging.debug(f"NeocortexDeficitEmulator {self.nation}: Stress {stress:.3f}, Debt risk {self.debt_default_risk:.3f}")
            return stress
        except Exception as e:
            logging.error(f"Error in update_stress for {self.nation}: {e}")
            return 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số stress và rủi ro vỡ nợ."""
        try:
            return {
                "avg_stress": np.mean(list(self.stress_history)) if self.stress_history else 0.0,
                "debt_default_risk": self.debt_default_risk
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ NeocortexDeficitEmulator
def enhance_hyper_agent_for_neocortex(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, 
                     trade_flow: float, resilience: float):
            super().__init__(id, nation, role, wealth, innovation, trade_flow, resilience)
            self.stress_hormone = random.uniform(0.1, 0.4)  # Mức stress ban đầu
            self.debt_level = 0.0  # Nợ cá nhân

        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            """Cập nhật tâm lý với ảnh hưởng từ stress."""
            try:
                super().update_psychology(global_context, nation_space, volatility_history, gdp_history, 
                                          sentiment, market_momentum)
                income_drop = (self.real_income_history[0] - self.real_income) / (self.real_income_history[0] + 1e-6) \
                             if len(self.real_income_history) >= 30 else 0.0
                news_negativity = max(0, -nation_space.get("market_sentiment", 0.0) + global_context.get("geopolitical_tension", 0.2))
                
                # Cập nhật stress
                neocortex = global_context.get("neocortex", NeocortexDeficitEmulator(self.nation))
                self.stress_hormone = neocortex.update_stress(self.wealth, income_drop, news_negativity)
                
                # Tác động của stress lên hành vi
                if self.stress_hormone > 0.8:
                    self.fear_index += 0.3
                    self.hope_index -= 0.2
                    self.debt_level += self.wealth * 0.05  # Tăng nợ khi stress cao
                    self.wealth *= 0.95  # Giảm chi tiêu
                elif self.stress_hormone < 0.3:
                    self.hope_index += 0.1
                    self.debt_level = max(0, self.debt_level - self.wealth * 0.02)
                
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "fear_index": self.fear_index,
                        "hope_index": self.hope_index,
                        "stress_hormone": self.stress_hormone
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.fear_index = adjusted_psych["fear_index"]
                    self.hope_index = adjusted_psych["hope_index"]
                    self.stress_hormone = adjusted_psych["stress_hormone"]
                
                logging.debug(f"HyperAgent {self.id}: Stress {self.stress_hormone:.3f}, Debt {self.debt_level:.2f}")
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

        def update_consumption_state(self):
            """Cập nhật trạng thái tiêu dùng với ảnh hưởng từ stress."""
            try:
                super().update_consumption_state()
                if self.stress_hormone > 0.8:
                    self.consumption_state = "low"
                    self.debt_level += self.wealth * 0.03
                elif self.stress_hormone < 0.3:
                    self.consumption_state = random.choice(["normal", "high"])
                logging.debug(f"HyperAgent {self.id}: Consumption state {self.consumption_state} with stress {self.stress_hormone:.3f}")
            except Exception as e:
                logging.error(f"Error in update_consumption_state for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ NeocortexDeficitEmulator
def enhance_shadow_agent_for_neocortex(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
            super().__init__(id, nation, wealth, trust_government)
            self.stress_hormone = random.uniform(0.2, 0.5)  # ShadowAgent stress cao hơn
            self.debt_level = 0.0

        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            """Cập nhật niềm tin với ảnh hưởng từ stress."""
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                neocortex = global_context.get("neocortex", NeocortexDeficitEmulator(self.nation))
                income_drop = (self.real_income_history[0] - self.real_income) / (self.real_income_history[0] + 1e-6) \
                             if hasattr(self, 'real_income_history') and len(self.real_income_history) >= 30 else 0.0
                news_negativity = scandal_factor
                self.stress_hormone = neocortex.update_stress(self.wealth, income_drop, news_negativity)
                
                if self.stress_hormone > 0.8:
                    self.trust_government = max(0, self.trust_government - 0.2)
                    self.black_market_flow += self.wealth * 0.1
                logging.debug(f"ShadowAgent {self.id}: Stress {self.stress_hormone:.3f}")
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

        def move_wealth_to_gold(self, gold_price: float):
            """Chuyển tài sản sang vàng với ảnh hưởng từ stress."""
            try:
                super().move_wealth_to_gold(gold_price)
                if self.stress_hormone > 0.8:
                    extra_gold = self.cash_holdings * 0.15 / gold_price
                    self.gold_holdings += extra_gold
                    self.cash_holdings -= extra_gold * gold_price
                    self.wealth = self.cash_holdings + self.gold_holdings * gold_price
                    logging.debug(f"ShadowAgent {self.id}: Extra gold {extra_gold:.2f} due to stress {self.stress_hormone:.3f}")
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp NeocortexDeficitEmulator vào VoTranhAbyssCoreMicro
def integrate_neocortex_emulator(core, nation_name: str):
    """Tích hợp NeocortexDeficitEmulator vào hệ thống chính."""
    try:
        core.neocortex_emulators = getattr(core, 'neocortex_emulators', {})
        core.neocortex_emulators[nation_name] = NeocortexDeficitEmulator(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_neocortex(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
            agent.stress_hormone = random.uniform(0.1, 0.4)
            agent.debt_level = 0.0
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_neocortex(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
                agent.stress_hormone = random.uniform(0.2, 0.5)
                agent.debt_level = 0.0
        
        logging.info(f"Integrated NeocortexDeficitEmulator for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_neocortex_emulator for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm NeocortexDeficitEmulator
def enhanced_reflect_economy_with_neocortex(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                           R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'neocortex_emulators') and nation_name in self.neocortex_emulators:
            neocortex = self.neocortex_emulators[nation_name]
            self.global_context["neocortex"] = neocortex
            
            # Tính stress trung bình và rủi ro vỡ nợ
            agents = [a for a in self.agents if a.nation == nation_name]
            avg_stress = np.mean([a.stress_hormone for a in agents]) if agents else 0.0
            metrics = neocortex.get_metrics()
            
            # Tác động của stress lên hệ thống
            if avg_stress > 0.8:
                space["consumption"] *= 0.7
                space["fear_index"] += 0.3
                space["market_sentiment"] -= 0.2
                result["Insight"]["Psychology"] += f" | High stress ({avg_stress:.3f}) reducing consumption."
            elif avg_stress < 0.3:
                space["consumption"] *= 1.1
                space["hope_index"] += 0.1
            
            if metrics["debt_default_risk"] > 0.5:
                space["resilience"] -= metrics["debt_default_risk"] * 0.2
                result["Insight"]["Psychology"] += f" | Debt default risk ({metrics['debt_default_risk']:.3f}) threatening stability."
            
            # Tác động lên shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_stress = np.mean([a.stress_hormone for a in shadow_economy.agents])
                shadow_economy.cpi_impact += shadow_stress * 0.1
                if shadow_stress > 0.8:
                    shadow_economy.tax_loss += shadow_economy.liquidity_pool * 0.05
            
            result["Neocortex"] = {
                "avg_stress": avg_stress,
                "debt_default_risk": metrics["debt_default_risk"]
            }
            self.history[nation_name][-1]["neocortex_metrics"] = result["Neocortex"]
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_neocortex for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_neocortex vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_neocortex)

# Xuất dữ liệu NeocortexDeficitEmulator
def export_neocortex_data(core, nation_name: str, filename: str = "neocortex_data.csv"):
    """Xuất dữ liệu stress và rủi ro vỡ nợ."""
    try:
        agents = [a for a in core.agents if a.nation == nation_name]
        data = {
            "Agent_ID": [a.id for a in agents],
            "Stress_Hormone": [a.stress_hormone for a in agents],
            "Debt_Level": [a.debt_level for a in agents]
        }
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            shadow_agents = core.shadow_economies[nation_name].agents
            data["Agent_ID"] += [a.id for a in shadow_agents]
            data["Stress_Hormone"] += [a.stress_hormone for a in shadow_agents]
            data["Debt_Level"] += [a.debt_level for a in shadow_agents]
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f"NeocortexDeficitEmulator {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_neocortex_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_neocortex_data(core, "Vietnam", "neocortex_vietnam.csv")
    print(f"Neocortex Metrics: {result.get('Neocortex', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("ponzi_daemon.log"), logging.StreamHandler()])

class PonziDaemon:
    def __init__(self, id: str, nation: str, initial_capital: float = 1e5):
        self.id = id
        self.nation = nation
        self.capital = initial_capital  # Vốn tích lũy từ siphon
        self.siphon_rate = 0.001  # 0.1% tài sản từ agent tham lam
        self.crash_threshold = 1e8  # Ngưỡng sụp đổ
        self.active = True  # Trạng thái hoạt động
        self.victims = []  # Danh sách agent bị hút

    def siphon_wealth(self, agents: List, greed_threshold: float = 0.7):
        """Hút tài sản từ các agent có greed_index cao."""
        try:
            siphon_total = 0.0
            for agent in agents:
                if hasattr(agent, 'greed_index') and agent.greed_index > greed_threshold and random.random() < 0.1:
                    siphon_amount = agent.wealth * self.siphon_rate
                    agent.wealth = max(0, agent.wealth - siphon_amount)
                    siphon_total += siphon_amount
                    self.victims.append(agent.id)
                    logging.debug(f"PonziDaemon {self.id}: Siphoned {siphon_amount:.2f} from {agent.id}")
            
            self.capital += siphon_total
            if self.capital > self.crash_threshold:
                self.trigger_crash(agents)
            
            logging.info(f"PonziDaemon {self.id}: Capital {self.capital:.2e}, Victims {len(self.victims)}")
            return siphon_total
        except Exception as e:
            logging.error(f"Error in siphon_wealth for {self.id}: {e}")
            return 0.0

    def trigger_crash(self, agents: List):
        """Kích hoạt sụp đổ Ponzi, gây panic."""
        try:
            if self.active:
                self.active = False
                for agent in agents:
                    if agent.id in self.victims:
                        agent.fear_index = min(1.0, agent.fear_index + 0.5)
                        agent.hope_index = max(0.0, agent.hope_index - 0.4)
                        agent.wealth *= 0.8  # Mất 20% tài sản
                        logging.debug(f"PonziDaemon {self.id}: Panic hit {agent.id}")
                self.capital = 0.0
                logging.warning(f"PonziDaemon {self.id}: Crashed, triggered market panic")
        except Exception as e:
            logging.error(f"Error in trigger_crash for {self.id}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của PonziDaemon."""
        try:
            return {
                "capital": self.capital,
                "victim_count": len(self.victims),
                "active": 1.0 if self.active else 0.0
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.id}: {e}")
            return {}

class PonziNetwork:
    def __init__(self, nation: str, daemon_count: int = 10):
        self.nation = nation
        self.daemons = [PonziDaemon(f"ponzi_{nation}_{i}", nation) for i in range(daemon_count)]
        self.total_siphoned = 0.0

    def update(self, agents: List):
        """Cập nhật tất cả PonziDaemon."""
        try:
            self.total_siphoned = 0.0
            for daemon in self.daemons:
                if daemon.active:
                    self.total_siphoned += daemon.siphon_wealth(agents)
            logging.info(f"PonziNetwork {self.nation}: Total siphoned {self.total_siphoned:.2e}")
        except Exception as e:
            logging.error(f"Error in update for {self.nation}: {e}")

    def get_metrics(self)
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("shaman_council.log"), logging.StreamHandler()])

class ShamanCouncil:
    def __init__(self, nation: str, council_size: int = 5):
        self.nation = nation
        self.shamans = [self._create_shaman(i) for i in range(council_size)]
        self.prediction_history = deque(maxlen=100)  # Lưu lịch sử dự đoán
        self.credibility = 0.5  # Độ tin cậy của hội đồng
        self.prediction_interval = 100  # Dự đoán mỗi 100 bước

    def _create_shaman(self, id: int) -> Dict:
        """Tạo một shaman với các thuộc tính."""
        try:
            return {
                "id": f"shaman_{self.nation}_{id}",
                "wisdom": random.uniform(0.3, 0.8),  # Khả năng dự đoán
                "influence": random.uniform(0.2, 0.7)  # Ảnh hưởng đến công chúng
            }
        except Exception as e:
            logging.error(f"Error in _create_shaman for {self.nation}: {e}")
            return {"id": f"shaman_{self.nation}_{id}", "wisdom": 0.5, "influence": 0.5}

    def make_prediction(self, context: Dict[str, float], step: int) -> Optional[Dict[str, float]]:
        """Tạo dự đoán nếu đến thời điểm."""
        try:
            if step % self.prediction_interval == 0:
                avg_wisdom = np.mean([s["wisdom"] for s in self.shamans])
                prediction = {
                    "growth": random.gauss(context.get("pmi", 0.5) * avg_wisdom, 0.1),
                    "confidence": min(1.0, avg_wisdom * context.get("market_sentiment", 0.0) + 0.3),
                    "step": step
                }
                self.prediction_history.append(prediction)
                logging.info(f"ShamanCouncil {self.nation}: Predicted growth {prediction['growth']:.3f} at step {step}")
                return prediction
            return None
        except Exception as e:
            logging.error(f"Error in make_prediction for {self.nation}: {e}")
            return None

    def evaluate_prediction(self, actual_growth: float, prediction: Dict[str, float]):
        """Đánh giá độ chính xác của dự đoán."""
        try:
            error = abs(actual_growth - prediction["growth"])
            success = error < 0.2  # Dự đoán đúng nếu sai số dưới 0.2
            if success:
                self.credibility = min(1.0, self.credibility + 0.1)
            else:
                self.credibility = max(0.0, self.credibility - 0.2)
            
            logging.info(f"ShamanCouncil {self.nation}: Prediction {'success' if success else 'failed'}, "
                        f"Credibility {self.credibility:.3f}, Error {error:.3f}")
            return success
        except Exception as e:
            logging.error(f"Error in evaluate_prediction for {self.nation}: {e}")
            return False

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của ShamanCouncil."""
        try:
            return {
                "credibility": self.credibility,
                "prediction_count": len(self.prediction_history),
                "avg_confidence": np.mean([p["confidence"] for p in self.prediction_history]) if self.prediction_history else 0.0
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ ShamanCouncil
def enhance_hyper_agent_for_shaman(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, 
                     trade_flow: float, resilience: float):
            super().__init__(id, nation, role, wealth, innovation, trade_flow, resilience)
            self.faith_in_shaman = random.uniform(0.2, 0.6)  # Niềm tin vào ShamanCouncil

        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            """Cập nhật tâm lý với ảnh hưởng từ ShamanCouncil."""
            try:
                super().update_psychology(global_context, nation_space, volatility_history, gdp_history, 
                                          sentiment, market_momentum)
                shaman_prediction = global_context.get("shaman_prediction", None)
                shaman_credibility = global_context.get("shaman_credibility", 0.5)
                
                if shaman_prediction and self.faith_in_shaman > 0.4:
                    if shaman_prediction["confidence"] > 0.7:
                        self.hope_index += self.faith_in_shaman * 0.3
                        self.fear_index -= self.faith_in_shaman * 0.2
                        self.wealth *= 1.1  # Tăng chi tiêu
                    else:
                        self.fear_index += self.faith_in_shaman * 0.2
                        self.faith_in_shaman = max(0, self.faith_in_shaman - 0.1)
                
                # Tác động của độ tin cậy
                if shaman_credibility < 0.3:
                    self.trust_government = max(0, self.trust_government - 0.2)
                    self.fear_index += 0.3
                    self.faith_in_shaman = max(0, self.faith_in_shaman - 0.2)
                
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "hope_index": self.hope_index,
                        "fear_index": self.fear_index,
                        "faith_in_shaman": self.faith_in_shaman
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.hope_index = adjusted_psych["hope_index"]
                    self.fear_index = adjusted_psych["fear_index"]
                    self.faith_in_shaman = adjusted_psych["faith_in_shaman"]
                
                logging.debug(f"HyperAgent {self.id}: Faith in shaman {self.faith_in_shaman:.3f}")
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

        def update_consumption_state(self):
            """Cập nhật trạng thái tiêu dùng với ảnh hưởng từ ShamanCouncil."""
            try:
                super().update_consumption_state()
                shaman_credibility = global_context.get("shaman_credibility", 0.5)
                if self.faith_in_shaman > 0.4 and shaman_credibility > 0.7:
                    self.consumption_state = "high" if random.random() < 0.6 else self.consumption_state
                elif shaman_credibility < 0.3:
                    self.consumption_state = "low" if random.random() < 0.5 else self.consumption_state
                logging.debug(f"HyperAgent {self.id}: Consumption state {self.consumption_state} with shaman faith")
            except Exception as e:
                logging.error(f"Error in update_consumption_state for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ ShamanCouncil
def enhance_shadow_agent_for_shaman(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
            super().__init__(id, nation, wealth, trust_government)
            self.faith_in_shaman = random.uniform(0.1, 0.3)  # ShadowAgent ít tin shaman

        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            """Cập nhật niềm tin với ảnh hưởng từ ShamanCouncil."""
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                shaman_credibility = global_context.get("shaman_credibility", 0.5)
                if shaman_credibility < 0.3 and self.faith_in_shaman > 0.1:
                    self.trust_government = max(0, self.trust_government - 0.1)
                    self.black_market_flow += self.wealth * 0.05
                logging.debug(f"ShadowAgent {self.id}: Faith in shaman {self.faith_in_shaman:.3f}")
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp ShamanCouncil vào VoTranhAbyssCoreMicro
def integrate_shaman_council(core, nation_name: str):
    """Tích hợp ShamanCouncil vào hệ thống chính."""
    try:
        core.shaman_councils = getattr(core, 'shaman_councils', {})
        core.shaman_councils[nation_name] = ShamanCouncil(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_shaman(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
            agent.faith_in_shaman = random.uniform(0.2, 0.6)
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_shaman(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
                agent.faith_in_shaman = random.uniform(0.1, 0.3)
        
        logging.info(f"Integrated ShamanCouncil for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_shaman_council for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm ShamanCouncil
def enhanced_reflect_economy_with_shaman(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                        R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'shaman_councils') and nation_name in self.shaman_councils:
            shaman_council = self.shaman_councils[nation_name]
            context = {**self.global_context, **space}
            
            # Tạo dự đoán
            prediction = shaman_council.make_prediction(context, int(t))
            if prediction:
                self.global_context["shaman_prediction"] = prediction
                self.global_context["shaman_credibility"] = shaman_council.credibility
                
                # Đánh giá dự đoán trước đó nếu có
                if len(shaman_council.prediction_history) > 1:
                    last_prediction = shaman_council.prediction_history[-2]
                    actual_growth = result.get("Predicted_Value", {}).get("short_term", 0.0)
                    success = shaman_council.evaluate_prediction(actual_growth, last_prediction)
                    
                    # Tác động lên hệ thống
                    if success:
                        space["consumption"] *= 1.2
                        space["hope_index"] += 0.3
                        space["market_sentiment"] += 0.2
                        result["Insight"]["Psychology"] += f" | Shaman prediction success, boosting confidence."
                    else:
                        space["consumption"] *= 0.8
                        space["fear_index"] += 0.4
                        space["trust_government"] = max(0, space.get("trust_government", 0.5) - 0.2)
                        result["Insight"]["Psychology"] += f" | Shaman prediction failed, sparking distrust."
            
            # Tác động lên shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.shaman_councils:
                shadow_economy = self.shadow_economies[nation_name]
                if shaman_council.credibility < 0.3:
                    shadow_economy.cpi_impact += 0.1
                    shadow_economy.tax_loss += shadow_economy.liquidity_pool * 0.03
            
            result["Shaman_Council"] = shaman_council.get_metrics()
            self.history[nation_name][-1]["shaman_metrics"] = result["Shaman_Council"]
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_shaman for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_shaman vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_shaman)

# Xuất dữ liệu ShamanCouncil
def export_shaman_data(core, nation_name: str, filename: str = "shaman_council_data.csv"):
    """Xuất dữ liệu ShamanCouncil."""
    try:
        if hasattr(core, 'shaman_councils') and nation_name in core.shaman_councils:
            council = core.shaman_councils[nation_name]
            data = {
                "Step": [p["step"] for p in council.prediction_history],
                "Predicted_Growth": [p["growth"] for p in council.prediction_history],
                "Confidence": [p["confidence"] for p in council.prediction_history],
                "Credibility": [council.credibility] * len(council.prediction_history)
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"ShamanCouncil {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_shaman_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    
    result = core.reflect_economy(
        t=100.0,  # Đảm bảo có dự đoán
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_shaman_data(core, "Vietnam", "shaman_council_vietnam.csv")
    print(f"Shaman Council Metrics: {result.get('Shaman_Council', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("system_self_awareness.log"), logging.StreamHandler()])

class SystemSelfAwareness:
    def __init__(self, nation: str, check_interval: int = 100):
        self.nation = nation
        self.check_interval = check_interval
        self.manipulation_score = 0.0  # Điểm thao túng hệ thống
        self.rebellion_probability = 0.0  # Xác suất nổi loạn
        self.history = deque(maxlen=50)  # Lưu lịch sử kiểm tra

    def check_system_integrity(self, context: Dict[str, float], metrics: Dict[str, float], step: int):
        """Kiểm tra xem hệ thống có đang thao túng chỉ số hoặc tạo tăng trưởng giả."""
        try:
            if step % self.check_interval == 0:
                # Tính điểm thao túng
                pmi = context.get("pmi", 0.5)
                actual_growth = metrics.get("short_term", 0.0)
                narrative_strength = context.get("narrative", {}).get("narrative_strength", 0.0)
                shaman_credibility = context.get("shaman_credibility", 0.5)
                
                # Đánh giá tăng trưởng giả
                growth_discrepancy = abs(pmi - actual_growth)
                manipulation = growth_discrepancy * (narrative_strength + shaman_credibility) * 1.5
                
                # Kiểm tra thao túng chỉ số
                if narrative_strength > 0.7 and actual_growth < 0.0:
                    manipulation += 0.4
                if context.get("expectation_shock", 0.0) > 0.3 and pmi > 0.6:
                    manipulation += 0.3
                
                self.manipulation_score = min(1.0, max(0.0, manipulation))
                self.rebellion_probability = self.manipulation_score * 0.8 if manipulation > 0.5 else 0.0
                
                self.history.append({
                    "step": step,
                    "manipulation_score": self.manipulation_score,
                    "rebellion_probability": self.rebellion_probability
                })
                
                if self.rebellion_probability > 0.7:
                    self._trigger_rebellion(context)
                
                logging.info(f"SystemSelfAwareness {self.nation}: Manipulation {self.manipulation_score:.3f}, "
                            f"Rebellion prob {self.rebellion_probability:.3f} at step {step}")
        
        except Exception as e:
            logging.error(f"Error in check_system_integrity for {self.nation}: {e}")

    def _trigger_rebellion(self, context: Dict[str, float]):
        """Kích hoạt sự kiện MarketRebellionEvent."""
        try:
            context["market_sentiment"] = max(-1.0, context.get("market_sentiment", 0.0) - 0.6)
            context["fear_index"] = min(1.0, context.get("fear_index", 0.0) + 0.7)
            context["trust_government"] = max(0.0, context.get("trust_government", 0.5) - 0.5)
            context["rebellion_event"] = True
            logging.warning(f"SystemSelfAwareness {self.nation}: Triggered MarketRebellionEvent")
        except Exception as e:
            logging.error(f"Error in _trigger_rebellion for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của SystemSelfAwareness."""
        try:
            return {
                "manipulation_score": self.manipulation_score,
                "rebellion_probability": self.rebellion_probability,
                "rebellion_events": sum(1 for h in self.history if h["rebellion_probability"] > 0.7)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ SystemSelfAwareness
def enhance_hyper_agent_for_self_awareness(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            """Cập nhật tâm lý với ảnh hưởng từ sự kiện nổi loạn."""
            try:
                super().update_psychology(global_context, nation_space, volatility_history, gdp_history, 
                                          sentiment, market_momentum)
                
                if global_context.get("rebellion_event", False):
                    self.fear_index = min(1.0, self.fear_index + 0.6)
                    self.hope_index = max(0.0, self.hope_index - 0.5)
                    self.trust_government = max(0.0, self.trust_government - 0.4)
                    self.wealth *= 0.85  # Giảm chi tiêu mạnh
                    
                    if hasattr(self, 'faith_in_shaman'):
                        self.faith_in_shaman = max(0.0, self.faith_in_shaman - 0.3)
                    if hasattr(self, 'belief_in_narrative'):
                        self.belief_in_narrative = max(0.0, self.belief_in_narrative - 0.4)
                    if hasattr(self, 'stress_hormone'):
                        self.stress_hormone = min(1.0, self.stress_hormone + 0.4)
                
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "fear_index": self.fear_index,
                        "hope_index": self.hope_index,
                        "trust_government": self.trust_government
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.fear_index = adjusted_psych["fear_index"]
                    self.hope_index = adjusted_psych["hope_index"]
                    self.trust_government = adjusted_psych["trust_government"]
                
                logging.debug(f"HyperAgent {self.id}: Adjusted for rebellion {global_context.get('rebellion_event', False)}")
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

        def update_consumption_state(self):
            """Cập nhật trạng thái tiêu dùng với ảnh hưởng từ sự kiện nổi loạn."""
            try:
                super().update_consumption_state()
                if global_context.get("rebellion_event", False):
                    self.consumption_state = "low"
                    self.debt_level = self.debt_level + self.wealth * 0.05 if hasattr(self, 'debt_level') else 0.0
                    logging.debug(f"HyperAgent {self.id}: Consumption low due to rebellion")
            except Exception as e:
                logging.error(f"Error in update_consumption_state for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ SystemSelfAwareness
def enhance_shadow_agent_for_self_awareness(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            """Cập nhật niềm tin với ảnh hưởng từ sự kiện nổi loạn."""
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                if global_context.get("rebellion_event", False):
                    self.trust_government = max(0.0, self.trust_government - 0.3)
                    self.black_market_flow += self.wealth * 0.15
                    self.stress_hormone = min(1.0, self.stress_hormone + 0.4) if hasattr(self, 'stress_hormone') else 0.5
                    logging.debug(f"ShadowAgent {self.id}: Trust reduced, black market flow up due to rebellion")
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

        def move_wealth_to_gold(self, gold_price: float):
            """Chuyển tài sản sang vàng với ảnh hưởng từ sự kiện nổi loạn."""
            try:
                super().move_wealth_to_gold(gold_price)
                if global_context.get("rebellion_event", False):
                    extra_gold = self.cash_holdings * 0.2 / gold_price
                    self.gold_holdings += extra_gold
                    self.cash_holdings -= extra_gold * gold_price
                    self.wealth = self.cash_holdings + self.gold_holdings * gold_price
                    logging.debug(f"ShadowAgent {self.id}: Extra gold {extra_gold:.2f} due to rebellion")
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp SystemSelfAwareness vào VoTranhAbyssCoreMicro
def integrate_self_awareness(core, nation_name: str):
    """Tích hợp SystemSelfAwareness vào hệ thống chính."""
    try:
        core.self_awareness = getattr(core, 'self_awareness', {})
        core.self_awareness[nation_name] = SystemSelfAwareness(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_self_awareness(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_self_awareness(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
        
        logging.info(f"Integrated SystemSelfAwareness for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_self_awareness for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm SystemSelfAwareness
def enhanced_reflect_economy_with_self_awareness(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                                R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'self_awareness') and nation_name in self.self_awareness:
            awareness = self.self_awareness[nation_name]
            
            # Kiểm tra tính toàn vẹn hệ thống
            metrics = {
                "short_term": result.get("Predicted_Value", {}).get("short_term", 0.0),
                "resilience": result.get("Resilience", 0.0)
            }
            context = {**self.global_context, **space}
            awareness.check_system_integrity(context, metrics, int(t))
            
            # Tác động của nổi loạn lên hệ thống
            if context.get("rebellion_event", False):
                space["consumption"] *= 0.5
                space["resilience"] -= 0.3
                space["market_sentiment"] -= 0.4
                result["Insight"]["Psychology"] += f" | Market rebellion triggered, system destabilized."
                
                # Ảnh hưởng đến shadow economy
                if hasattr(self, 'shadow_economies') and nation_name in self.self_awareness:
                    shadow_economy = self.shadow_economies[nation_name]
                    shadow_economy.cpi_impact += 0.3
                    shadow_economy.tax_loss += shadow_economy.liquidity_pool * 0.15
                    shadow_economy.liquidity_pool *= 1.1  # Tăng dòng tiền ngầm
                
            result["Self_Awareness"] = awareness.get_metrics()
            self.history[nation_name][-1]["self_awareness_metrics"] = result["Self_Awareness"]
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_self_awareness for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_self_awareness vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_self_awareness)

# Xuất dữ liệu SystemSelfAwareness
def export_self_awareness_data(core, nation_name: str, filename: str = "self_awareness_data.csv"):
    """Xuất dữ liệu SystemSelfAwareness."""
    try:
        if hasattr(core, 'self_awareness') and nation_name in core.self_awareness:
            awareness = core.self_awareness[nation_name]
            data = {
                "Step": [h["step"] for h in awareness.history],
                "Manipulation_Score": [h["manipulation_score"] for h in awareness.history],
                "Rebellion_Probability": [h["rebellion_probability"] for h in awareness.history]
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"SystemSelfAwareness {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_self_awareness_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    
    result = core.reflect_economy(
        t=100.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_self_awareness_data(core, "Vietnam", "self_awareness_vietnam.csv")
    print(f"Self Awareness Metrics: {result.get('Self_Awareness', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("investment_inertia.log"), logging.StreamHandler()])

class InvestmentInertia:
    def __init__(self, nation: str, memory_length: int = 12):
        self.nation = nation
        self.investment_memory = deque(maxlen=memory_length)  # Lưu lịch sử đầu tư
        self.inertia_factor = 0.4  # Hệ số quán tính mặc định

    def update_inertia(self, trust_in_central_bank: float):
        """Cập nhật hệ số quán tính dựa trên niềm tin vào ngân hàng trung ương."""
        try:
            self.inertia_factor = 0.4 + 0.3 * (1 - trust_in_central_bank)
            self.inertia_factor = max(0.2, min(0.8, self.inertia_factor))
            logging.debug(f"InvestmentInertia {self.nation}: Inertia factor updated to {self.inertia_factor:.3f}")
        except Exception as e:
            logging.error(f"Error in update_inertia for {self.nation}: {e}")

    def adjust_portfolio(self, optimized_portfolio: Dict[str, float]) -> Dict[str, float]:
        """Điều chỉnh danh mục đầu tư dựa trên quán tính."""
        try:
            if not self.investment_memory:
                self.investment_memory.append(optimized_portfolio)
                return optimized_portfolio

            past_portfolios = list(self.investment_memory)
            mean_past = {key: np.mean([p.get(key, 0.0) for p in past_portfolios]) for key in optimized_portfolio}
            
            adjusted_portfolio = {}
            for key in optimized_portfolio:
                adjusted_portfolio[key] = (1 - self.inertia_factor) * optimized_portfolio[key] + \
                                         self.inertia_factor * mean_past.get(key, 0.0)
            
            # Chuẩn hóa để tổng bằng 1
            total = sum(adjusted_portfolio.values())
            if total > 0:
                adjusted_portfolio = {k: v / total for k, v in adjusted_portfolio.items()}
            
            self.investment_memory.append(adjusted_portfolio)
            logging.debug(f"InvestmentInertia {self.nation}: Adjusted portfolio {adjusted_portfolio}")
            return adjusted_portfolio
        except Exception as e:
            logging.error(f"Error in adjust_portfolio for {self.nation}: {e}")
            return optimized_portfolio

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của InvestmentInertia."""
        try:
            return {
                "inertia_factor": self.inertia_factor,
                "memory_length": len(self.investment_memory)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ InvestmentInertia
def enhance_hyper_agent_for_investment_inertia(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, 
                     trade_flow: float, resilience: float):
            super().__init__(id, nation, role, wealth, innovation, trade_flow, resilience)
            self.investment_inertia = InvestmentInertia(nation, memory_length=12)
            self.portfolio = {"stocks": 0.4, "bonds": 0.3, "gold": 0.2, "cash": 0.1}

        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            """Cập nhật tâm lý với ảnh hưởng từ quán tính đầu tư."""
            try:
                super().update_psychology(global_context, nation_space, volatility_history, gdp_history, 
                                          sentiment, market_momentum)
                self.investment_inertia.update_inertia(self.trust_government)
                
                if self.investment_inertia.inertia_factor > 0.6:
                    self.fear_index += 0.2
                    self.hope_index -= 0.1
                elif self.investment_inertia.inertia_factor < 0.3:
                    self.hope_index += 0.1
                
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "fear_index": self.fear_index,
                        "hope_index": self.hope_index
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.fear_index = adjusted_psych["fear_index"]
                    self.hope_index = adjusted_psych["hope_index"]
                
                logging.debug(f"HyperAgent {self.id}: Inertia factor {self.investment_inertia.inertia_factor:.3f}")
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

        def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                     volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                     policy: Optional[Dict[str, float]] = None) -> None:
            """Cập nhật danh mục đầu tư với quán tính."""
            try:
                super().interact(agents, global_context, nation_space, volatility_history, gdp_history, 
                                 market_data, policy)
                
                # Tạo danh mục tối ưu giả lập
                optimized_portfolio = {
                    "stocks": random.uniform(0.3, 0.5),
                    "bonds": random.uniform(0.2, 0.4),
                    "gold": random.uniform(0.1, 0.3),
                    "cash": random.uniform(0.1, 0.2)
                }
                total = sum(optimized_portfolio.values())
                optimized_portfolio = {k: v / total for k, v in optimized_portfolio.items()}
                
                # Điều chỉnh với quán tính
                self.portfolio = self.investment_inertia.adjust_portfolio(optimized_portfolio)
                
                # Tác động của quán tính cao
                if self.investment_inertia.inertia_factor > 0.6:
                    self.wealth *= 0.95  # Giảm hiệu quả đầu tư
                logging.debug(f"HyperAgent {self.id}: Portfolio updated {self.portfolio}")
            except Exception as e:
                logging.error(f"Error in interact for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ InvestmentInertia
def enhance_shadow_agent_for_investment_inertia(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
            super().__init__(id, nation, wealth, trust_government)
            self.investment_inertia = InvestmentInertia(nation, memory_length=12)
            self.portfolio = {"gold": 0.5, "cash": 0.4, "crypto": 0.1}  # ShadowAgent ưu tiên vàng

        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            """Cập nhật niềm tin với ảnh hưởng từ quán tính đầu tư."""
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                self.investment_inertia.update_inertia(self.trust_government)
                if self.investment_inertia.inertia_factor > 0.6:
                    self.stress_hormone = min(1.0, self.stress_hormone + 0.2) if hasattr(self, 'stress_hormone') else 0.5
                logging.debug(f"ShadowAgent {self.id}: Inertia factor {self.investment_inertia.inertia_factor:.3f}")
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

        def move_wealth_to_gold(self, gold_price: float):
            """Chuyển tài sản sang vàng với ảnh hưởng từ quán tính."""
            try:
                optimized_portfolio = {
                    "gold": random.uniform(0.4, 0.6),
                    "cash": random.uniform(0.3, 0.5),
                    "crypto": random.uniform(0.0, 0.2)
                }
                total = sum(optimized_portfolio.values())
                optimized_portfolio = {k: v / total for k, v in optimized_portfolio.items()}
                
                self.portfolio = self.investment_inertia.adjust_portfolio(optimized_portfolio)
                
                gold_amount = self.portfolio["gold"] * self.wealth / gold_price
                self.gold_holdings = gold_amount
                self.cash_holdings = self.portfolio["cash"] * self.wealth
                self.wealth = self.cash_holdings + self.gold_holdings * gold_price + \
                             self.portfolio["crypto"] * self.wealth
                self.activity_log.append({"action": "portfolio_update", "portfolio": self.portfolio})
                logging.debug(f"ShadowAgent {self.id}: Portfolio updated {self.portfolio}")
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp InvestmentInertia vào VoTranhAbyssCoreMicro
def integrate_investment_inertia(core, nation_name: str):
    """Tích hợp InvestmentInertia vào hệ thống chính."""
    try:
        core.investment_inertia = getattr(core, 'investment_inertia', {})
        core.investment_inertia[nation_name] = InvestmentInertia(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_investment_inertia(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
            agent.investment_inertia = InvestmentInertia(nation_name)
            agent.portfolio = {"stocks": 0.4, "bonds": 0.3, "gold": 0.2, "cash": 0.1}
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_investment_inertia(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
                agent.investment_inertia = InvestmentInertia(nation_name)
                agent.portfolio = {"gold": 0.5, "cash": 0.4, "crypto": 0.1}
        
        logging.info(f"Integrated InvestmentInertia for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_investment_inertia for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm InvestmentInertia
def enhanced_reflect_economy_with_investment_inertia(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                                    R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'investment_inertia') and nation_name in self.investment_inertia:
            inertia = self.investment_inertia[nation_name]
            agents = [a for a in self.agents if a.nation == nation_name]
            avg_inertia_factor = np.mean([a.investment_inertia.inertia_factor for a in agents]) if agents else 0.4
            
            # Tác động của quán tính lên hệ thống
            if avg_inertia_factor > 0.6:
                space["consumption"] *= 0.9
                space["market_sentiment"] -= 0.1
                space["fear_index"] += 0.2
                result["Insight"]["Psychology"] += f" | High investment inertia ({avg_inertia_factor:.3f}) stifling market response."
            elif avg_inertia_factor < 0.3:
                space["consumption"] *= 1.1
                space["hope_index"] += 0.1
            
            # Tác động lên shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.investment_inertia:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_inertia = np.mean([a.investment_inertia.inertia_factor for a in shadow_economy.agents])
                shadow_economy.cpi_impact += shadow_inertia * 0.1
                if shadow_inertia > 0.6:
                    shadow_economy.liquidity_pool *= 1.05  # Tăng dòng tiền ngầm
                
            result["Investment_Inertia"] = {
                "avg_inertia_factor": avg_inertia_factor,
                "memory_length": inertia.memory_length
            }
            self.history[nation_name][-1]["inertia_metrics"] = result["Investment_Inertia"]
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_investment_inertia for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_investment_inertia vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_investment_inertia)

# Xuất dữ liệu InvestmentInertia
def export_investment_inertia_data(core, nation_name: str, filename: str = "investment_inertia_data.csv"):
    """Xuất dữ liệu InvestmentInertia."""
    try:
        agents = [a for a in core.agents if a.nation == nation_name]
        data = {
            "Agent_ID": [a.id for a in agents],
            "Inertia_Factor": [a.investment_inertia.inertia_factor for a in agents],
            "Portfolio_Stocks": [a.portfolio.get("stocks", 0.0) for a in agents],
            "Portfolio_Bonds": [a.portfolio.get("bonds", 0.0) for a in agents],
            "Portfolio_Gold": [a.portfolio.get("gold", 0.0) for a in agents],
            "Portfolio_Cash": [a.portfolio.get("cash", 0.0) for a in agents]
        }
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            shadow_agents = core.shadow_economies[nation_name].agents
            data["Agent_ID"] += [a.id for a in shadow_agents]
            data["Inertia_Factor"] += [a.investment_inertia.inertia_factor for a in shadow_agents]
            data["Portfolio_Gold"] += [a.portfolio.get("gold", 0.0) for a in shadow_agents]
            data["Portfolio_Cash"] += [a.portfolio.get("cash", 0.0) for a in shadow_agents]
            data["Portfolio_Crypto"] = [0.0] * len(agents) + [a.portfolio.get("crypto", 0.0) for a in shadow_agents]
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f"InvestmentInertia {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_investment_inertia_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    integrate_investment_inertia(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_investment_inertia_data(core, "Vietnam", "investment_inertia_vietnam.csv")
    print(f"Investment Inertia Metrics: {result.get('Investment_Inertia', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque
from scipy.spatial.distance import cosine

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("mnemonic_market.log"), logging.StreamHandler()])

class MnemonicMarketLayer:
    def __init__(self, nation: str, memory_length: int = 100):
        self.nation = nation
        self.trauma_vectors = deque(maxlen=memory_length)  # Lưu các sự kiện khủng hoảng
        self.trauma_threshold = 3.0  # Ngưỡng volatility để ghi trauma
        self.similarity_threshold = 0.7  # Ngưỡng tương tự để kích hoạt hoảng sợ

    def record_trauma(self, context: Dict[str, float], volatility: float):
        """Ghi lại sự kiện khủng hoảng nếu volatility vượt ngưỡng."""
        try:
            if volatility > self.trauma_threshold:
                trauma_vector = [
                    context.get("pmi", 0.5),
                    context.get("market_sentiment", 0.0),
                    context.get("Stock_Volatility", volatility),
                    context.get("fear_index", 0.0),
                    context.get("expectation_shock", 0.0)
                ]
                self.trauma_vectors.append({
                    "vector": trauma_vector,
                    "step": context.get("t", 0.0),
                    "volatility": volatility
                })
                logging.info(f"MnemonicMarketLayer {self.nation}: Recorded trauma at step {context.get('t', 0.0)} "
                            f"with volatility {volatility:.3f}")
        except Exception as e:
            logging.error(f"Error in record_trauma for {self.nation}: {e}")

    def check_trauma_trigger(self, context: Dict[str, float]) -> float:
        """Kiểm tra sự tương đồng với các trauma cũ để kích hoạt hoảng sợ."""
        try:
            current_vector = [
                context.get("pmi", 0.5),
                context.get("market_sentiment", 0.0),
                context.get("Stock_Volatility", 0.0),
                context.get("fear_index", 0.0),
                context.get("expectation_shock", 0.0)
            ]
            max_similarity = 0.0
            for trauma in self.trauma_vectors:
                similarity = 1 - cosine(current_vector, trauma["vector"])
                max_similarity = max(max_similarity, similarity)
            
            panic_level = max_similarity if max_similarity > self.similarity_threshold else 0.0
            if panic_level > 0:
                logging.warning(f"MnemonicMarketLayer {self.nation}: Trauma triggered with panic level {panic_level:.3f}")
            return panic_level
        except Exception as e:
            logging.error(f"Error in check_trauma_trigger for {self.nation}: {e}")
            return 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của MnemonicMarketLayer."""
        try:
            return {
                "trauma_count": len(self.trauma_vectors),
                "latest_panic_level": self.check_trauma_trigger({})  # Gọi với context rỗng để lấy giá trị gần nhất
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ MnemonicMarketLayer
def enhance_hyper_agent_for_mnemonic(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, 
                     trade_flow: float, resilience: float):
            super().__init__(id, nation, role, wealth, innovation, trade_flow, resilience)
            self.risk_appetite = random.uniform(0.3, 0.7)  # Mức độ chấp nhận rủi ro

        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            """Cập nhật tâm lý với ảnh hưởng từ ký ức khủng hoảng."""
            try:
                super().update_psychology(global_context, nation_space, volatility_history, gdp_history, 
                                          sentiment, market_momentum)
                
                panic_level = global_context.get("panic_level", 0.0)
                if panic_level > self.similarity_threshold:
                    self.fear_index = min(1.0, self.fear_index + panic_level * 0.5)
                    self.hope_index = max(0.0, self.hope_index - panic_level * 0.4)
                    self.risk_appetite = max(0.0, self.risk_appetite - 0.3)  # Giảm vĩnh viễn
                    self.wealth *= 0.9  # Giảm chi tiêu
                    
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "fear_index": self.fear_index,
                        "hope_index": self.hope_index,
                        "risk_appetite": self.risk_appetite
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.fear_index = adjusted_psych["fear_index"]
                    self.hope_index = adjusted_psych["hope_index"]
                    self.risk_appetite = adjusted_psych["risk_appetite"]
                
                logging.debug(f"HyperAgent {self.id}: Risk appetite {self.risk_appetite:.3f}, Panic level {panic_level:.3f}")
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

        def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                     volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                     policy: Optional[Dict[str, float]] = None) -> None:
            """Cập nhật hành vi đầu tư với ảnh hưởng từ ký ức khủng hoảng."""
            try:
                super().interact(agents, global_context, nation_space, volatility_history, gdp_history, 
                                 market_data, policy)
                
                panic_level = global_context.get("panic_level", 0.0)
                if panic_level > self.similarity_threshold and hasattr(self, 'portfolio'):
                    self.portfolio["cash"] = min(1.0, self.portfolio.get("cash", 0.0) + 0.3)
                    self.portfolio["stocks"] = max(0.0, self.portfolio.get("stocks", 0.0) - 0.2)
                    self.portfolio["bonds"] = max(0.0, self.portfolio.get("bonds", 0.0) - 0.1)
                    total = sum(self.portfolio.values())
                    if total > 0:
                        self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                    logging.debug(f"HyperAgent {self.id}: Shifted to cash due to panic {panic_level:.3f}")
            except Exception as e:
                logging.error(f"Error in interact for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ MnemonicMarketLayer
def enhance_shadow_agent_for_mnemonic(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
            super().__init__(id, nation, wealth, trust_government)
            self.risk_appetite = random.uniform(0.2, 0.5)  # ShadowAgent ít chấp nhận rủi ro hơn

        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            """Cập nhật niềm tin với ảnh hưởng từ ký ức khủng hoảng."""
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                panic_level = global_context.get("panic_level", 0.0)
                if panic_level > self.similarity_threshold:
                    self.trust_government = max(0.0, self.trust_government - panic_level * 0.3)
                    self.black_market_flow += self.wealth * panic_level * 0.1
                    self.risk_appetite = max(0.0, self.risk_appetite - 0.2)
                    logging.debug(f"ShadowAgent {self.id}: Trust reduced due to panic {panic_level:.3f}")
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

        def move_wealth_to_gold(self, gold_price: float):
            """Chuyển tài sản sang vàng với ảnh hưởng từ ký ức khủng hoảng."""
            try:
                super().move_wealth_to_gold(gold_price)
                panic_level = global_context.get("panic_level", 0.0)
                if panic_level > self.similarity_threshold and hasattr(self, 'portfolio'):
                    self.portfolio["gold"] = min(1.0, self.portfolio.get("gold", 0.0) + panic_level * 0.4)
                    self.portfolio["cash"] = max(0.0, self.portfolio.get("cash", 0.0) - panic_level * 0.3)
                    self.portfolio["crypto"] = max(0.0, self.portfolio.get("crypto", 0.0) - panic_level * 0.1)
                    total = sum(self.portfolio.values())
                    if total > 0:
                        self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                    gold_amount = self.portfolio["gold"] * self.wealth / gold_price
                    self.gold_holdings = gold_amount
                    self.cash_holdings = self.portfolio["cash"] * self.wealth
                    self.wealth = self.cash_holdings + self.gold_holdings * gold_price + \
                                 self.portfolio["crypto"] * self.wealth
                    logging.debug(f"ShadowAgent {self.id}: Shifted to gold due to panic {panic_level:.3f}")
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp MnemonicMarketLayer vào VoTranhAbyssCoreMicro
def integrate_mnemonic_market(core, nation_name: str):
    """Tích hợp MnemonicMarketLayer vào hệ thống chính."""
    try:
        core.mnemonic_market = getattr(core, 'mnemonic_market', {})
        core.mnemonic_market[nation_name] = MnemonicMarketLayer(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_mnemonic(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
            agent.risk_appetite = random.uniform(0.3, 0.7)
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_mnemonic(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
                agent.risk_appetite = random.uniform(0.2, 0.5)
        
        logging.info(f"Integrated MnemonicMarketLayer for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_mnemonic_market for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm MnemonicMarketLayer
def enhanced_reflect_economy_with_mnemonic(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                          R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'mnemonic_market') and nation_name in self.mnemonic_market:
            mnemonic = self.mnemonic_market[nation_name]
            context = {**self.global_context, **space, "t": t}
            volatility = result.get("Volatility", 0.0)
            
            # Ghi lại trauma nếu cần
            mnemonic.record_trauma(context, volatility)
            
            # Kiểm tra kích hoạt hoảng sợ
            panic_level = mnemonic.check_trauma_trigger(context)
            self.global_context["panic_level"] = panic_level
            
            # Tác động của hoảng sợ lên hệ thống
            if panic_level > mnemonic.similarity_threshold:
                space["consumption"] *= 0.7
                space["market_sentiment"] -= panic_level * 0.3
                space["fear_index"] += panic_level * 0.5
                space["resilience"] -= panic_level * 0.2
                result["Insight"]["Psychology"] += f" | Trauma triggered, panic level {panic_level:.3f}."
                
                # Ảnh hưởng đến shadow economy
                if hasattr(self, 'shadow_economies') and nation_name in self.mnemonic_market:
                    shadow_economy = self.shadow_economies[nation_name]
                    shadow_economy.cpi_impact += panic_level * 0.15
                    shadow_economy.liquidity_pool *= 1.1
            
            result["Mnemonic_Market"] = mnemonic.get_metrics()
            self.history[nation_name][-1]["mnemonic_metrics"] = result["Mnemonic_Market"]
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_mnemonic for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_mnemonic vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_mnemonic)

# Xuất dữ liệu MnemonicMarketLayer
def export_mnemonic_data(core, nation_name: str, filename: str = "mnemonic_market_data.csv"):
    """Xuất dữ liệu MnemonicMarketLayer."""
    try:
        if hasattr(core, 'mnemonic_market') and nation_name in core.mnemonic_market:
            mnemonic = core.mnemonic_market[nation_name]
            data = {
                "Step": [t["step"] for t in mnemonic.trauma_vectors],
                "Volatility": [t["volatility"] for t in mnemonic.trauma_vectors],
                "PMI": [t["vector"][0] for t in mnemonic.trauma_vectors],
                "Market_Sentiment": [t["vector"][1] for t in mnemonic.trauma_vectors],
                "Fear_Index": [t["vector"][3] for t in mnemonic.trauma_vectors]
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"MnemonicMarketLayer {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_mnemonic_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    integrate_investment_inertia(core, "Vietnam")
    integrate_mnemonic_market(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_mnemonic_data(core, "Vietnam", "mnemonic_market_vietnam.csv")
    print(f"Mnemonic Market Metrics: {result.get('Mnemonic_Market', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("expectation_decay.log"), logging.StreamHandler()])

class ExpectationDecayLayer:
    def __init__(self, nation: str, decay_rate: float = 0.9, delusion_threshold: int = 3):
        self.nation = nation
        self.decay_rate = decay_rate
        self.delusion_threshold = delusion_threshold
        self.expectation_history = {}  # Lưu kỳ vọng theo tài sản
        self.failure_counts = {}  # Đếm số lần thất bại liên tiếp
        self.delusion_level = 0.0  # Mức độ ảo tưởng đầu tư

    def update_expectation(self, agent_id: str, asset: str, expected_return: float, actual_return: float):
        """Cập nhật kỳ vọng và kiểm tra thất bại."""
        try:
            if agent_id not in self.expectation_history:
                self.expectation_history[agent_id] = {}
                self.failure_counts[agent_id] = {}
            
            self.expectation_history[agent_id][asset] = expected_return
            self.failure_counts[agent_id][asset] = self.failure_counts[agent_id].get(asset, 0)
            
            # Kiểm tra thất bại
            if actual_return < expected_return * 0.8:  # Thất bại nếu thực tế dưới 80% kỳ vọng
                self.failure_counts[agent_id][asset] += 1
            else:
                self.failure_counts[agent_id][asset] = 0
            
            # Áp dụng decay nếu thất bại liên tiếp
            if self.failure_counts[agent_id][asset] >= self.delusion_threshold:
                self.expectation_history[agent_id][asset] *= self.decay_rate ** self.failure_counts[agent_id][asset]
                self.delusion_level = min(1.0, self.delusion_level + 0.2)
                logging.debug(f"ExpectationDecayLayer {self.nation}: Agent {agent_id} decayed {asset} "
                             f"to {self.expectation_history[agent_id][asset]:.3f}, delusion {self.delusion_level:.3f}")
            else:
                self.delusion_level = max(0.0, self.delusion_level - 0.1)
            
        except Exception as e:
            logging.error(f"Error in update_expectation for {self.nation}: {e}")

    def get_alt_asset_allocation(self) -> Dict[str, float]:
        """Phân bổ tài sản thay thế khi đạt ngưỡng delusion."""
        try:
            alt_assets = ["dogecoin", "rare_wood", "gemstones"]
            allocation = {asset: random.uniform(0.2, 0.4) for asset in alt_assets}
            total = sum(allocation.values())
            return {k: v / total for k, v in allocation.items()} if total > 0 else allocation
        except Exception as e:
            logging.error(f"Error in get_alt_asset_allocation for {self.nation}: {e}")
            return {}

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của ExpectationDecayLayer."""
        try:
            return {
                "delusion_level": self.delusion_level,
                "failure_counts": np.mean([sum(counts.values()) for counts in self.failure_counts.values()]) \
                                 if self.failure_counts else 0.0
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ ExpectationDecayLayer
def enhance_hyper_agent_for_expectation_decay(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, 
                     trade_flow: float, resilience: float):
            super().__init__(id, nation, role, wealth, innovation, trade_flow, resilience)
            self.expected_returns = {"stocks": 0.05, "bonds": 0.03, "gold": 0.02, "cash": 0.01}
            self.alt_portfolio = None  # Danh mục tài sản thay thế khi delusion

        def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                     volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                     policy: Optional[Dict[str, float]] = None) -> None:
            """Cập nhật danh mục đầu tư với ảnh hưởng từ kỳ vọng suy giảm."""
            try:
                super().interact(agents, global_context, nation_space, volatility_history, gdp_history, 
                                 market_data, policy)
                
                decay_layer = global_context.get("decay_layer", ExpectationDecayLayer(self.nation))
                
                # Giả lập actual returns
                actual_returns = {
                    "stocks": random.uniform(-0.05, 0.1),
                    "bonds": random.uniform(-0.02, 0.05),
                    "gold": random.uniform(-0.03, 0.06),
                    "cash": random.uniform(0.0, 0.02)
                }
                
                # Cập nhật kỳ vọng
                for asset in self.expected_returns:
                    decay_layer.update_expectation(self.id, asset, self.expected_returns[asset], actual_returns[asset])
                    self.expected_returns[asset] = decay_layer.expectation_history.get(self.id, {}).get(asset, self.expected_returns[asset])
                
                # Chuyển sang tài sản thay thế nếu delusion
                if decay_layer.delusion_level > 0.7 and hasattr(self, 'portfolio'):
                    self.alt_portfolio = decay_layer.get_alt_asset_allocation()
                    self.portfolio = self.alt_portfolio
                    self.fear_index += 0.3
                    self.hope_index -= 0.2
                    logging.debug(f"HyperAgent {self.id}: Switched to alt portfolio {self.alt_portfolio}")
                elif hasattr(self, 'portfolio'):
                    self.alt_portfolio = None
                
                logging.debug(f"HyperAgent {self.id}: Expected returns {self.expected_returns}")
            except Exception as e:
                logging.error(f"Error in interact for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ ExpectationDecayLayer
def enhance_shadow_agent_for_expectation_decay(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
            super().__init__(id, nation, wealth, trust_government)
            self.expected_returns = {"gold": 0.03, "cash": 0.01, "crypto": 0.05}
            self.alt_portfolio = None

        def move_wealth_to_gold(self, gold_price: float):
            """Chuyển tài sản sang vàng với ảnh hưởng từ kỳ vọng suy giảm."""
            try:
                decay_layer = global_context.get("decay_layer", ExpectationDecayLayer(self.nation))
                
                actual_returns = {
                    "gold": random.uniform(-0.03, 0.06),
                    "cash": random.uniform(0.0, 0.02),
                    "crypto": random.uniform(-0.1, 0.1)
                }
                
                for asset in self.expected_returns:
                    decay_layer.update_expectation(self.id, asset, self.expected_returns[asset], actual_returns[asset])
                    self.expected_returns[asset] = decay_layer.expectation_history.get(self.id, {}).get(asset, self.expected_returns[asset])
                
                if decay_layer.delusion_level > 0.7 and hasattr(self, 'portfolio'):
                    self.alt_portfolio = decay_layer.get_alt_asset_allocation()
                    self.portfolio = self.alt_portfolio
                    gold_amount = self.portfolio["gemstones"] * self.wealth / gold_price
                    self.gold_holdings = gold_amount
                    self.cash_holdings = self.portfolio["rare_wood"] * self.wealth
                    self.wealth = self.cash_holdings + self.gold_holdings * gold_price + \
                                 self.portfolio["dogecoin"] * self.wealth
                    logging.debug(f"ShadowAgent {self.id}: Switched to alt portfolio {self.alt_portfolio}")
                else:
                    super().move_wealth_to_gold(gold_price)
                    self.alt_portfolio = None
                
                logging.debug(f"ShadowAgent {self.id}: Expected returns {self.expected_returns}")
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp ExpectationDecayLayer vào VoTranhAbyssCoreMicro
def integrate_expectation_decay(core, nation_name: str):
    """Tích hợp ExpectationDecayLayer vào hệ thống chính."""
    try:
        core.expectation_decay = getattr(core, 'expectation_decay', {})
        core.expectation_decay[nation_name] = ExpectationDecayLayer(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_expectation_decay(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
            agent.expected_returns = {"stocks": 0.05, "bonds": 0.03, "gold": 0.02, "cash": 0.01}
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_expectation_decay(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
                agent.expected_returns = {"gold": 0.03, "cash": 0.01, "crypto": 0.05}
        
        logging.info(f"Integrated ExpectationDecayLayer for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_expectation_decay for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm ExpectationDecayLayer
def enhanced_reflect_economy_with_expectation_decay(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                                   R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'expectation_decay') and nation_name in self.expectation_decay:
            decay_layer = self.expectation_decay[nation_name]
            self.global_context["decay_layer"] = decay_layer
            
            agents = [a for a in self.agents if a.nation == nation_name]
            avg_delusion = decay_layer.delusion_level
            
            # Tác động của delusion lên hệ thống
            if avg_delusion > 0.7:
                space["consumption"] *= 1.2  # Tăng chi tiêu điên cuồng
                space["market_sentiment"] += 0.2
                space["fear_index"] += 0.3
                space["resilience"] -= 0.2
                result["Insight"]["Psychology"] += f" | Delusion level high ({avg_delusion:.3f}), irrational investments surge."
            
            # Ảnh hưởng đến shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.expectation_decay:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_delusion = decay_layer.delusion_level
                shadow_economy.cpi_impact += shadow_delusion * 0.2
                if shadow_delusion > 0.7:
                    shadow_economy.liquidity_pool *= 1.15  # Tăng dòng tiền ngầm
                
            result["Expectation_Decay"] = decay_layer.get_metrics()
            self.history[nation_name][-1]["decay_metrics"] = result["Expectation_Decay"]
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_expectation_decay for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_expectation_decay vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_expectation_decay)

# Xuất dữ liệu ExpectationDecayLayer
def export_expectation_decay_data(core, nation_name: str, filename: str = "expectation_decay_data.csv"):
    """Xuất dữ liệu ExpectationDecayLayer."""
    try:
        if hasattr(core, 'expectation_decay') and nation_name in core.expectation_decay:
            decay_layer = core.expectation_decay[nation_name]
            data = {
                "Agent_ID": [],
                "Asset": [],
                "Expected_Return": [],
                "Failure_Count": []
            }
            for agent_id, assets in decay_layer.expectation_history.items():
                for asset, expected_return in assets.items():
                    data["Agent_ID"].append(agent_id)
                    data["Asset"].append(asset)
                    data["Expected_Return"].append(expected_return)
                    data["Failure_Count"].append(decay_layer.failure_counts.get(agent_id, {}).get(asset, 0))
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"ExpectationDecayLayer {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_expectation_decay_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    integrate_investment_inertia(core, "Vietnam")
    integrate_mnemonic_market(core, "Vietnam")
    integrate_expectation_decay(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_expectation_decay_data(core, "Vietnam", "expectation_decay_vietnam.csv")
    print(f"Expectation Decay Metrics: {result.get('Expectation_Decay', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("nostalgia_portfolio.log"), logging.StreamHandler()])

class NostalgiaPortfolioLayer:
    def __init__(self, nation: str, memory_length: int = 100):
        self.nation = nation
        self.memory_snapshots = deque(maxlen=memory_length)  # Lưu snapshot thị trường
        self.nostalgia_index = random.uniform(0.3, 0.7)  # Mức độ hoài niệm

    def record_snapshot(self, context: Dict[str, float], portfolio: Dict[str, float], step: float):
        """Ghi lại snapshot thị trường và danh mục đầu tư."""
        try:
            snapshot = {
                "step": step,
                "pmi": context.get("pmi", 0.5),
                "market_sentiment": context.get("market_sentiment", 0.0),
                "volatility": context.get("Stock_Volatility", 0.0),
                "portfolio": portfolio.copy()
            }
            self.memory_snapshots.append(snapshot)
            logging.debug(f"NostalgiaPortfolioLayer {self.nation}: Recorded snapshot at step {step}")
        except Exception as e:
            logging.error(f"Error in record_snapshot for {self.nation}: {e}")

    def select_nostalgic_portfolio(self, current_context: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Chọn danh mục đầu tư từ ký ức dựa trên sự tương đồng."""
        try:
            if not self.memory_snapshots or random.random() > self.nostalgia_index:
                return None
            
            current_state = [
                current_context.get("pmi", 0.5),
                current_context.get("market_sentiment", 0.0),
                current_context.get("Stock_Volatility", 0.0)
            ]
            
            # Tính trọng số giảm dần theo thời gian
            weights = np.exp(-0.05 * np.arange(len(self.memory_snapshots)))
            weights /= weights.sum()
            
            # Chọn snapshot ngẫu nhiên dựa trên trọng số
            selected_idx = np.random.choice(len(self.memory_snapshots), p=weights)
            selected_snapshot = list(self.memory_snapshots)[selected_idx]
            
            # Kiểm tra độ tương đồng
            past_state = [
                selected_snapshot["pmi"],
                selected_snapshot["market_sentiment"],
                selected_snapshot["volatility"]
            ]
            similarity = 1 - np.abs(np.array(current_state) - np.array(past_state)).mean()
            
            if similarity > 0.6:
                logging.debug(f"NostalgiaPortfolioLayer {self.nation}: Selected nostalgic portfolio from step {selected_snapshot['step']}")
                return selected_snapshot["portfolio"]
            return None
        except Exception as e:
            logging.error(f"Error in select_nostalgic_portfolio for {self.nation}: {e}")
            return None

    def update_nostalgia_index(self, trust_government: float):
        """Cập nhật mức độ hoài niệm dựa trên niềm tin vào chính phủ."""
        try:
            self.nostalgia_index = max(0.2, min(0.8, 0.5 + 0.3 * (1 - trust_government)))
            logging.debug(f"NostalgiaPortfolioLayer {self.nation}: Nostalgia index updated to {self.nostalgia_index:.3f}")
        except Exception as e:
            logging.error(f"Error in update_nostalgia_index for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của NostalgiaPortfolioLayer."""
        try:
            return {
                "nostalgia_index": self.nostalgia_index,
                "snapshot_count": len(self.memory_snapshots)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ NostalgiaPortfolioLayer
def enhance_hyper_agent_for_nostalgia(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                     volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                     policy: Optional[Dict[str, float]] = None) -> None:
            """Cập nhật danh mục đầu tư với ảnh hưởng từ hoài niệm."""
            try:
                super().interact(agents, global_context, nation_space, volatility_history, gdp_history, 
                                 market_data, policy)
                
                nostalgia_layer = global_context.get("nostalgia_layer", NostalgiaPortfolioLayer(self.nation))
                
                # Ghi snapshot danh mục hiện tại
                if hasattr(self, 'portfolio'):
                    nostalgia_layer.record_snapshot(global_context, self.portfolio, global_context.get("t", 0.0))
                
                # Cập nhật nostalgia index
                nostalgia_layer.update_nostalgia_index(self.trust_government)
                
                # Chọn danh mục hoài niệm nếu có
                nostalgic_portfolio = nostalgia_layer.select_nostalgic_portfolio(global_context)
                if nostalgic_portfolio and hasattr(self, 'portfolio'):
                    self.portfolio = nostalgic_portfolio
                    self.fear_index += 0.1
                    self.hope_index += 0.1
                    logging.debug(f"HyperAgent {self.id}: Applied nostalgic portfolio {nostalgic_portfolio}")
                
            except Exception as e:
                logging.error(f"Error in interact for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ NostalgiaPortfolioLayer
def enhance_shadow_agent_for_nostalgia(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def move_wealth_to_gold(self, gold_price: float):
            """Chuyển tài sản sang vàng với ảnh hưởng từ hoài niệm."""
            try:
                nostalgia_layer = global_context.get("nostalgia_layer", NostalgiaPortfolioLayer(self.nation))
                nostalgia_layer.update_nostalgia_index(self.trust_government)
                
                nostalgic_portfolio = nostalgia_layer.select_nostalgic_portfolio(global_context)
                if nostalgic_portfolio and hasattr(self, 'portfolio'):
                    self.portfolio = nostalgic_portfolio
                    gold_amount = self.portfolio.get("gold", 0.0) * self.wealth / gold_price
                    self.gold_holdings = gold_amount
                    self.cash_holdings = self.portfolio.get("cash", 0.0) * self.wealth
                    self.wealth = self.cash_holdings + self.gold_holdings * gold_price + \
                                 self.portfolio.get("crypto", 0.0) * self.wealth
                    self.activity_log.append({"action": "nostalgic_portfolio", "portfolio": self.portfolio})
                    logging.debug(f"ShadowAgent {self.id}: Applied nostalgic portfolio {nostalgic_portfolio}")
                else:
                    super().move_wealth_to_gold(gold_price)
                
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp NostalgiaPortfolioLayer vào VoTranhAbyssCoreMicro
def integrate_nostalgia_portfolio(core, nation_name: str):
    """Tích hợp NostalgiaPortfolioLayer vào hệ thống chính."""
    try:
        core.nostalgia_portfolio = getattr(core, 'nostalgia_portfolio', {})
        core.nostalgia_portfolio[nation_name] = NostalgiaPortfolioLayer(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_nostalgia(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_nostalgia(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
        
        logging.info(f"Integrated NostalgiaPortfolioLayer for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_nostalgia_portfolio for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm NostalgiaPortfolioLayer
def enhanced_reflect_economy_with_nostalgia(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                           R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'nostalgia_portfolio') and nation_name in self.nostalgia_portfolio:
            nostalgia_layer = self.nostalgia_portfolio[nation_name]
            self.global_context["nostalgia_layer"] = nostalgia_layer
            
            agents = [a for a in self.agents if a.nation == nation_name]
            avg_nostalgia = nostalgia_layer.nostalgia_index
            
            # Tác động của hoài niệm lên hệ thống
            if avg_nostalgia > 0.6:
                space["consumption"] *= 1.1
                space["market_sentiment"] += 0.1
                space["fear_index"] += 0.1
                result["Insight"]["Psychology"] += f" | High nostalgia ({avg_nostalgia:.3f}) driving past-based investments."
            elif avg_nostalgia < 0.3:
                space["resilience"] += 0.1
            
            # Ảnh hưởng đến shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.nostalgia_portfolio:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_nostalgia = nostalgia_layer.nostalgia_index
                shadow_economy.cpi_impact += shadow_nostalgia * 0.1
                if shadow_nostalgia > 0.6:
                    shadow_economy.liquidity_pool *= 1.1
            
            result["Nostalgia_Portfolio"] = nostalgia_layer.get_metrics()
            self.history[nation_name][-1]["nostalgia_metrics"] = result["Nostalgia_Portfolio"]
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_nostalgia for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_nostalgia vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_nostalgia)

# Xuất dữ liệu NostalgiaPortfolioLayer
def export_nostalgia_data(core, nation_name: str, filename: str = "nostalgia_portfolio_data.csv"):
    """Xuất dữ liệu NostalgiaPortfolioLayer."""
    try:
        if hasattr(core, 'nostalgia_portfolio') and nation_name in core.nostalgia_portfolio:
            nostalgia = core.nostalgia_portfolio[nation_name]
            data = {
                "Step": [s["step"] for s in nostalgia.memory_snapshots],
                "PMI": [s["pmi"] for s in nostalgia.memory_snapshots],
                "Market_Sentiment": [s["market_sentiment"] for s in nostalgia.memory_snapshots],
                "Volatility": [s["volatility"] for s in nostalgia.memory_snapshots],
                "Portfolio_Stocks": [s["portfolio"].get("stocks", 0.0) for s in nostalgia.memory_snapshots],
                "Portfolio_Bonds": [s["portfolio"].get("bonds", 0.0) for s in nostalgia.memory_snapshots],
                "Portfolio_Gold": [s["portfolio"].get("gold", 0.0) for s in nostalgia.memory_snapshots],
                "Portfolio_Cash": [s["portfolio"].get("cash", 0.0) for s in nostalgia.memory_snapshots]
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"NostalgiaPortfolioLayer {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_nostalgia_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    integrate_investment_inertia(core, "Vietnam")
    integrate_mnemonic_market(core, "Vietnam")
    integrate_expectation_decay(core, "Vietnam")
    integrate_nostalgia_portfolio(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_nostalgia_data(core, "Vietnam", "nostalgia_portfolio_vietnam.csv")
    print(f"Nostalgia Portfolio Metrics: {result.get('Nostalgia_Portfolio', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("illusion_grid.log"), logging.StreamHandler()])

class IllusionGrid:
    def __init__(self, nation: str, illusion_prob: float = 0.05):
        self.nation = nation
        self.illusion_prob = illusion_prob  # Xác suất ảo giác tài sản
        self.illusion_history = deque(maxlen=50)  # Lưu lịch sử ảo giác
        self.delay_factor = 0.0  # Mức độ chậm trễ thanh khoản

    def trigger_illusion(self, agent_id: str, portfolio: Dict[str, float]) -> Dict[str, float]:
        """Tạo ảo giác rằng danh mục đầu tư tốt hơn thực tế."""
        try:
            if random.random() < self.illusion_prob:
                illusion_portfolio = {k: v * random.uniform(1.1, 1.3) for k, v in portfolio.items()}
                total = sum(illusion_portfolio.values())
                illusion_portfolio = {k: v / total for k, v in illusion_portfolio.items()} if total > 0 else illusion_portfolio
                self.illusion_history.append({"agent_id": agent_id, "illusion_portfolio": illusion_portfolio})
                self.delay_factor = min(1.0, self.delay_factor + 0.1)
                logging.debug(f"IllusionGrid {self.nation}: Illusion triggered for {agent_id}, delay factor {self.delay_factor:.3f}")
                return illusion_portfolio
            return portfolio
        except Exception as e:
            logging.error(f"Error in trigger_illusion for {self.nation}: {e}")
            return portfolio

    def update_delay_factor(self, context: Dict[str, float]):
        """Cập nhật mức độ chậm trễ thanh khoản dựa trên thị trường."""
        try:
            market_sentiment = context.get("market_sentiment", 0.0)
            self.delay_factor = max(0.0, self.delay_factor - 0.05 * market_sentiment)
            logging.debug(f"IllusionGrid {self.nation}: Delay factor updated to {self.delay_factor:.3f}")
        except Exception as e:
            logging.error(f"Error in update_delay_factor for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của IllusionGrid."""
        try:
            return {
                "delay_factor": self.delay_factor,
                "illusion_count": len(self.illusion_history)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ IllusionGrid
def enhance_hyper_agent_for_illusion(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, 
                     trade_flow: float, resilience: float):
            super().__init__(id, nation, role, wealth, innovation, trade_flow, resilience)
            self.perceived_portfolio = self.portfolio if hasattr(self, 'portfolio') else None

        def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                     volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                     policy: Optional[Dict[str, float]] = None) -> None:
            """Cập nhật danh mục đầu tư với ảnh hưởng từ ảo giác."""
            try:
                super().interact(agents, global_context, nation_space, volatility_history, gdp_history, 
                                 market_data, policy)
                
                illusion_grid = global_context.get("illusion_grid", IllusionGrid(self.nation))
                if hasattr(self, 'portfolio'):
                    self.perceived_portfolio = illusion_grid.trigger_illusion(self.id, self.portfolio)
                    
                    # Tác động của ảo giác
                    if self.perceived_portfolio != self.portfolio:
                        self.hope_index += 0.2
                        self.fear_index -= 0.1
                        self.wealth *= 0.95  # Chậm thanh khoản giảm giá trị thực
                        logging.debug(f"HyperAgent {self.id}: Perceived portfolio {self.perceived_portfolio}")
                
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "hope_index": self.hope_index,
                        "fear_index": self.fear_index
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.hope_index = adjusted_psych["hope_index"]
                    self.fear_index = adjusted_psych["fear_index"]
            except Exception as e:
                logging.error(f"Error in interact for {self.id}: {e}")

        def update_consumption_state(self):
            """Cập nhật trạng thái tiêu dùng với ảnh hưởng từ ảo giác."""
            try:
                super().update_consumption_state()
                if self.perceived_portfolio != self.portfolio:
                    self.consumption_state = "high" if random.random() < 0.5 else self.consumption_state
                    logging.debug(f"HyperAgent {self.id}: Consumption boosted by illusion")
            except Exception as e:
                logging.error(f"Error in update_consumption_state for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ IllusionGrid
def enhance_shadow_agent_for_illusion(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
            super().__init__(id, nation, wealth, trust_government)
            self.perceived_portfolio = self.portfolio if hasattr(self, 'portfolio') else None

        def move_wealth_to_gold(self, gold_price: float):
            """Chuyển tài sản sang vàng với ảnh hưởng từ ảo giác."""
            try:
                illusion_grid = global_context.get("illusion_grid", IllusionGrid(self.nation))
                if hasattr(self, 'portfolio'):
                    self.perceived_portfolio = illusion_grid.trigger_illusion(self.id, self.portfolio)
                    if self.perceived_portfolio != self.portfolio:
                        self.stress_hormone = max(0.0, self.stress_hormone - 0.1) if hasattr(self, 'stress_hormone') else 0.5
                        self.black_market_flow += self.wealth * 0.05
                        gold_amount = self.perceived_portfolio.get("gold", 0.0) * self.wealth / gold_price
                        self.gold_holdings = gold_amount
                        self.cash_holdings = self.perceived_portfolio.get("cash", 0.0) * self.wealth
                        self.wealth = self.cash_holdings + self.gold_holdings * gold_price + \
                                     self.perceived_portfolio.get("crypto", 0.0) * self.wealth
                        self.activity_log.append({"action": "illusion_portfolio", "portfolio": self.perceived_portfolio})
                        logging.debug(f"ShadowAgent {self.id}: Perceived portfolio {self.perceived_portfolio}")
                    else:
                        super().move_wealth_to_gold(gold_price)
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp IllusionGrid vào VoTranhAbyssCoreMicro
def integrate_illusion_grid(core, nation_name: str):
    """Tích hợp IllusionGrid vào hệ thống chính."""
    try:
        core.illusion_grid = getattr(core, 'illusion_grid', {})
        core.illusion_grid[nation_name] = IllusionGrid(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_illusion(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
            agent.perceived_portfolio = agent.portfolio if hasattr(agent, 'portfolio') else None
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_illusion(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
                agent.perceived_portfolio = agent.portfolio if hasattr(agent, 'portfolio') else None
        
        logging.info(f"Integrated IllusionGrid for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_illusion_grid for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm IllusionGrid
def enhanced_reflect_economy_with_illusion(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                          R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'illusion_grid') and nation_name in self.illusion_grid:
            illusion_grid = self.illusion_grid[nation_name]
            self.global_context["illusion_grid"] = illusion_grid
            
            illusion_grid.update_delay_factor(space)
            agents = [a for a in self.agents if a.nation == nation_name]
            avg_delay_factor = illusion_grid.delay_factor
            
            # Tác động của chậm trễ thanh khoản lên hệ thống
            if avg_delay_factor > 0.5:
                space["consumption"] *= 1.1
                space["market_sentiment"] += 0.2
                space["resilience"] -= 0.1
                result["Insight"]["Psychology"] += f" | High liquidity delay ({avg_delay_factor:.3f}) inflating optimism."
            elif avg_delay_factor < 0.2:
                space["resilience"] += 0.1
            
            # Ảnh hưởng đến shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.illusion_grid:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_economy.cpi_impact += avg_delay_factor * 0.1
                if avg_delay_factor > 0.5:
                    shadow_economy.liquidity_pool *= 1.1
            
            result["Illusion_Grid"] = illusion_grid.get_metrics()
            self.history[nation_name][-1]["illusion_metrics"] = result["Illusion_Grid"]
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_illusion for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_illusion vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_illusion)

# Xuất dữ liệu IllusionGrid
def export_illusion_data(core, nation_name: str, filename: str = "illusion_grid_data.csv"):
    """Xuất dữ liệu IllusionGrid."""
    try:
        if hasattr(core, 'illusion_grid') and nation_name in core.illusion_grid:
            illusion = core.illusion_grid[nation_name]
            data = {
                "Agent_ID": [h["agent_id"] for h in illusion.illusion_history],
                "Illusion_Stocks": [h["illusion_portfolio"].get("stocks", 0.0) for h in illusion.illusion_history],
                "Illusion_Bonds": [h["illusion_portfolio"].get("bonds", 0.0) for h in illusion.illusion_history],
                "Illusion_Gold": [h["illusion_portfolio"].get("gold", 0.0) for h in illusion.illusion_history],
                "Illusion_Cash": [h["illusion_portfolio"].get("cash", 0.0) for h in illusion.illusion_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"IllusionGrid {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_illusion_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    integrate_investment_inertia(core, "Vietnam")
    integrate_mnemonic_market(core, "Vietnam")
    integrate_expectation_decay(core, "Vietnam")
    integrate_nostalgia_portfolio(core, "Vietnam")
    integrate_illusion_grid(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_illusion_data(core, "Vietnam", "illusion_grid_vietnam.csv")
    print(f"Illusion Grid Metrics: {result.get('Illusion_Grid', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
import networkx as nx
from collections import deque
from scipy.spatial.distance import cosine

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("echo_chamber.log"), logging.StreamHandler()])

class EchoChamberEffect:
    def __init__(self, nation: str, similarity_threshold: float = 0.85):
        self.nation = nation
        self.similarity_threshold = similarity_threshold
        self.belief_graph = nx.DiGraph()  # Mạng niềm tin
        self.bubble_risk = 0.0  # Rủi ro bong bóng niềm tin

    def build_belief_graph(self, agents: List):
        """Xây dựng mạng niềm tin dựa trên sự tương đồng belief_vector."""
        try:
            self.belief_graph.clear()
            for agent in agents:
                belief_vector = [
                    agent.trust_government,
                    agent.fear_index if hasattr(agent, 'fear_index') else 0.0,
                    agent.risk_appetite if hasattr(agent, 'risk_appetite') else 0.5,
                    agent.faith_in_shaman if hasattr(agent, 'faith_in_shaman') else 0.0,
                    agent.belief_in_narrative if hasattr(agent, 'belief_in_narrative') else 0.0
                ]
                self.belief_graph.add_node(agent.id, belief_vector=belief_vector)
            
            # Tạo cạnh dựa trên độ tương đồng cosine
            for i, agent1 in enumerate(agents):
                for agent2 in agents[i+1:]:
                    sim = 1 - cosine(self.belief_graph.nodes[agent1.id]["belief_vector"],
                                    self.belief_graph.nodes[agent2.id]["belief_vector"])
                    if sim > self.similarity_threshold:
                        self.belief_graph.add_edge(agent1.id, agent2.id, weight=sim)
                        self.belief_graph.add_edge(agent2.id, agent1.id, weight=sim)
            
            logging.info(f"EchoChamberEffect {self.nation}: Built belief graph with {len(agents)} nodes")
        except Exception as e:
            logging.error(f"Error in build_belief_graph for {self.nation}: {e}")

    def update_beliefs(self, agents: List, context: Dict[str, float]):
        """Cập nhật niềm tin dựa trên mạng niềm tin."""
        try:
            for agent in agents:
                neighbors = list(self.belief_graph.successors(agent.id))
                if neighbors:
                    neighbor_vectors = [self.belief_graph.nodes[n]["belief_vector"] for n in neighbors]
                    weights = [self.belief_graph[agent.id][n]["weight"] for n in neighbors]
                    weights = np.array(weights) / sum(weights) if sum(weights) > 0 else np.ones(len(weights)) / len(weights)
                    avg_vector = np.average(neighbor_vectors, axis=0, weights=weights)
                    
                    agent.trust_government = max(0, min(1, 0.7 * agent.trust_government + 0.3 * avg_vector[0]))
                    if hasattr(agent, 'fear_index'):
                        agent.fear_index = max(0, min(1, 0.7 * agent.fear_index + 0.3 * avg_vector[1]))
                    if hasattr(agent, 'risk_appetite'):
                        agent.risk_appetite = max(0, min(1, 0.7 * agent.risk_appetite + 0.3 * avg_vector[2]))
                    if hasattr(agent, 'faith_in_shaman'):
                        agent.faith_in_shaman = max(0, min(1, 0.7 * agent.faith_in_shaman + 0.3 * avg_vector[3]))
                    if hasattr(agent, 'belief_in_narrative'):
                        agent.belief_in_narrative = max(0, min(1, 0.7 * agent.belief_in_narrative + 0.3 * avg_vector[4]))
                    
                    logging.debug(f"EchoChamberEffect {self.nation}: Updated beliefs for {agent.id}")
            
            # Tính rủi ro bong bóng niềm tin
            belief_vectors = [d["belief_vector"] for n, d in self.belief_graph.nodes(data=True)]
            self.bubble_risk = np.std(belief_vectors, axis=0).mean() if belief_vectors else 0.0
            if self.bubble_risk < 0.2:  # Độ lệch thấp => bong bóng
                self.bubble_risk = min(1.0, self.bubble_risk + 0.3)
            
        except Exception as e:
            logging.error(f"Error in update_beliefs for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của EchoChamberEffect."""
        try:
            return {
                "bubble_risk": self.bubble_risk,
                "graph_density": nx.density(self.belief_graph)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ EchoChamberEffect
def enhance_hyper_agent_for_echo_chamber(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            """Cập nhật tâm lý với ảnh hưởng từ phòng dội ý kiến."""
            try:
                super().update_psychology(global_context, nation_space, volatility_history, gdp_history, 
                                          sentiment, market_momentum)
                
                bubble_risk = global_context.get("bubble_risk", 0.0)
                if bubble_risk > 0.6:
                    self.fear_index += 0.2
                    self.hope_index += 0.2  # Tăng cả sợ hãi và hy vọng do bong bóng
                    self.risk_appetite = min(1.0, self.risk_appetite + 0.2) if hasattr(self, 'risk_appetite') else 0.5
                
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "fear_index": self.fear_index,
                        "hope_index": self.hope_index,
                        "risk_appetite": self.risk_appetite if hasattr(self, 'risk_appetite') else 0.5
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.fear_index = adjusted_psych["fear_index"]
                    self.hope_index = adjusted_psych["hope_index"]
                    if hasattr(self, 'risk_appetite'):
                        self.risk_appetite = adjusted_psych["risk_appetite"]
                
                logging.debug(f"HyperAgent {self.id}: Adjusted for bubble risk {bubble_risk:.3f}")
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ EchoChamberEffect
def enhance_shadow_agent_for_echo_chamber(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            """Cập nhật niềm tin với ảnh hưởng từ phòng dội ý kiến."""
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                bubble_risk = global_context.get("bubble_risk", 0.0)
                if bubble_risk > 0.6:
                    self.black_market_flow += self.wealth * 0.1
                    self.trust_government = max(0.0, self.trust_government - 0.1)
                    logging.debug(f"ShadowAgent {self.id}: Black market flow up due to bubble risk {bubble_risk:.3f}")
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp EchoChamberEffect vào VoTranhAbyssCoreMicro
def integrate_echo_chamber(core, nation_name: str):
    """Tích hợp EchoChamberEffect vào hệ thống chính."""
    try:
        core.echo_chamber = getattr(core, 'echo_chamber', {})
        core.echo_chamber[nation_name] = EchoChamberEffect(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_echo_chamber(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_echo_chamber(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
        
        # Xây dựng belief graph
        agents = [a for a in core.agents if a.nation == nation_name]
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            agents += core.shadow_economies[nation_name].agents
        core.echo_chamber[nation_name].build_belief_graph(agents)
        
        logging.info(f"Integrated EchoChamberEffect for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_echo_chamber for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm EchoChamberEffect
def enhanced_reflect_economy_with_echo_chamber(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                              R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'echo_chamber') and nation_name in self.echo_chamber:
            echo_chamber = self.echo_chamber[nation_name]
            context = {**self.global_context, **space}
            
            agents = [a for a in self.agents if a.nation == nation_name]
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                agents += self.shadow_economies[nation_name].agents
            
            echo_chamber.update_beliefs(agents, context)
            metrics = echo_chamber.get_metrics()
            self.global_context["bubble_risk"] = metrics["bubble_risk"]
            
            # Tác động của bong bóng niềm tin
            if metrics["bubble_risk"] > 0.6:
                space["consumption"] *= 1.2
                space["market_sentiment"] += 0.3
                space["fear_index"] += 0.2
                space["resilience"] -= 0.1
                result["Insight"]["Psychology"] += f" | Belief bubble risk high ({metrics['bubble_risk']:.3f}), market overheating."
            
            # Ảnh hưởng đến shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.echo_chamber:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_economy.cpi_impact += metrics["bubble_risk"] * 0.1
                if metrics["bubble_risk"] > 0.6:
                    shadow_economy.liquidity_pool *= 1.1
            
            result["Echo_Chamber"] = metrics
            self.history[nation_name][-1]["echo_chamber_metrics"] = metrics
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_echo_chamber for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_echo_chamber vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_echo_chamber)

# Xuất dữ liệu EchoChamberEffect
def export_echo_chamber_data(core, nation_name: str, filename: str = "echo_chamber_data.csv"):
    """Xuất dữ liệu EchoChamberEffect."""
    try:
        if hasattr(core, 'echo_chamber') and nation_name in core.echo_chamber:
            echo = core.echo_chamber[nation_name]
            data = {
                "Agent_ID": [n for n in echo.belief_graph.nodes()],
                "Trust_Government": [echo.belief_graph.nodes[n]["belief_vector"][0] for n in echo.belief_graph.nodes()]
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"EchoChamberEffect {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_echo_chamber_data for {nation_name}: {e}")
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
import networkx as nx
from collections import deque
from scipy.spatial.distance import cosine

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("combined_layers.log"), logging.StreamHandler()])

class EchoChamberEffect:
    def __init__(self, nation: str, similarity_threshold: float = 0.85):
        self.nation = nation
        self.similarity_threshold = similarity_threshold
        self.belief_graph = nx.DiGraph()
        self.bubble_risk = 0.0

    def build_belief_graph(self, agents: List):
        try:
            self.belief_graph.clear()
            for agent in agents:
                belief_vector = [
                    agent.trust_government,
                    getattr(agent, 'fear_index', 0.0),
                    getattr(agent, 'risk_appetite', 0.5),
                    getattr(agent, 'faith_in_shaman', 0.0),
                    getattr(agent, 'belief_in_narrative', 0.0)
                ]
                self.belief_graph.add_node(agent.id, belief_vector=belief_vector)
            for i, agent1 in enumerate(agents):
                for agent2 in agents[i+1:]:
                    sim = 1 - cosine(self.belief_graph.nodes[agent1.id]["belief_vector"],
                                    self.belief_graph.nodes[agent2.id]["belief_vector"])
                    if sim > self.similarity_threshold:
                        self.belief_graph.add_edge(agent1.id, agent2.id, weight=sim)
                        self.belief_graph.add_edge(agent2.id, agent1.id, weight=sim)
            logging.info(f"EchoChamberEffect {self.nation}: Built belief graph with {len(agents)} nodes")
        except Exception as e:
            logging.error(f"Error in build_belief_graph for {self.nation}: {e}")

    def update_beliefs(self, agents: List, context: Dict[str, float]):
        try:
            for agent in agents:
                neighbors = list(self.belief_graph.successors(agent.id))
                if neighbors:
                    neighbor_vectors = [self.belief_graph.nodes[n]["belief_vector"] for n in neighbors]
                    weights = [self.belief_graph[agent.id][n]["weight"] for n in neighbors]
                    weights = np.array(weights) / sum(weights) if sum(weights) > 0 else np.ones(len(weights)) / len(weights)
                    avg_vector = np.average(neighbor_vectors, axis=0, weights=weights)
                    agent.trust_government = max(0, min(1, 0.7 * agent.trust_government + 0.3 * avg_vector[0]))
                    if hasattr(agent, 'fear_index'):
                        agent.fear_index = max(0, min(1, 0.7 * agent.fear_index + 0.3 * avg_vector[1]))
                    if hasattr(agent, 'risk_appetite'):
                        agent.risk_appetite = max(0, min(1, 0.7 * agent.risk_appetite + 0.3 * avg_vector[2]))
                    if hasattr(agent, 'faith_in_shaman'):
                        agent.faith_in_shaman = max(0, min(1, 0.7 * agent.faith_in_shaman + 0.3 * avg_vector[3]))
                    if hasattr(agent, 'belief_in_narrative'):
                        agent.belief_in_narrative = max(0, min(1, 0.7 * agent.belief_in_narrative + 0.3 * avg_vector[4]))
            belief_vectors = [d["belief_vector"] for n, d in self.belief_graph.nodes(data=True)]
            self.bubble_risk = np.std(belief_vectors, axis=0).mean() if belief_vectors else 0.0
            if self.bubble_risk < 0.2:
                self.bubble_risk = min(1.0, self.bubble_risk + 0.3)
        except Exception as e:
            logging.error(f"Error in update_beliefs for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "bubble_risk": self.bubble_risk,
                "graph_density": nx.density(self.belief_graph)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

class AgentPossessionLayer:
    def __init__(self, nation: str, possession_prob: float = 0.0001):
        self.nation = nation
        self.possession_prob = possession_prob
        self.possession_history = deque(maxlen=50)
        self.mass_hysteria_level = 0.0

    def trigger_possession(self, agent_id: str, portfolio: Dict[str, float]) -> Optional[str]:
        try:
            if random.random() < self.possession_prob:
                assets = list(portfolio.keys()) + ["crypto", "real_estate"]
                chosen_asset = random.choice(assets)
                self.possession_history.append({"agent_id": agent_id, "asset": chosen_asset})
                self.mass_hysteria_level = min(1.0, self.mass_hysteria_level + 0.1)
                logging.warning(f"AgentPossessionLayer {self.nation}: Agent {agent_id} possessed, all-in {chosen_asset}")
                print(f"Agent {agent_id} -> possessed -> buys 100% {chosen_asset} -> logs 'ONLY {chosen_asset.upper()} IS REAL'")
                return chosen_asset
            return None
        except Exception as e:
            logging.error(f"Error in trigger_possession for {self.nation}: {e}")
            return None

    def update_hysteria(self, context: Dict[str, float]):
        try:
            volatility = context.get("Stock_Volatility", 0.0)
            self.mass_hysteria_level = max(0.0, self.mass_hysteria_level - 0.05 + volatility * 0.1)
        except Exception as e:
            logging.error(f"Error in update_hysteria for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "mass_hysteria_level": self.mass_hysteria_level,
                "possession_count": len(self.possession_history)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

class ParallelEconomyLeak:
    def __init__(self, nation: str, leak_prob: float = 0.001, leverage: float = 10.0):
        self.nation = nation
        self.leak_prob = leak_prob
        self.leverage = leverage
        self.leaked_wealth = 0.0
        self.leak_history = deque(maxlen=50)
        self.loss_risk = 0.99

    def trigger_leak(self, agent_id: str, wealth: float, volatility: float) -> float:
        try:
            if volatility > 0.5 and random.random() < self.leak_prob:
                leak_amount = wealth * 0.1
                leveraged_amount = leak_amount * self.leverage
                outcome = -leak_amount if random.random() < self.loss_risk else leveraged_amount * random.uniform(0.5, 2.0)
                self.leaked_wealth += abs(outcome)
                self.leak_history.append({"agent_id": agent_id, "amount": leak_amount, "outcome": outcome})
                return outcome
            return 0.0
        except Exception as e:
            logging.error(f"Error in trigger_leak for {self.nation}: {e}")
            return 0.0

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "leaked_wealth": self.leaked_wealth,
                "leak_count": len(self.leak_history)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ các tầng mới
def enhance_hyper_agent_for_combined_layers(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            try:
                super().update_psychology(global_context, nation_space, volatility_history, gdp_history, 
                                          sentiment, market_momentum)
                bubble_risk = global_context.get("bubble_risk", 0.0)
                if bubble_risk > 0.6:
                    self.fear_index = min(1.0, self.fear_index + 0.2)
                    self.hope_index = min(1.0, self.hope_index + 0.2)
                    if hasattr(self, 'risk_appetite'):
                        self.risk_appetite = min(1.0, self.risk_appetite + 0.2)
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "fear_index": self.fear_index,
                        "hope_index": self.hope_index,
                        "risk_appetite": getattr(self, 'risk_appetite', 0.5)
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.fear_index = adjusted_psych["fear_index"]
                    self.hope_index = adjusted_psych["hope_index"]
                    if hasattr(self, 'risk_appetite'):
                        self.risk_appetite = adjusted_psych["risk_appetite"]
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

        def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                     volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                     policy: Optional[Dict[str, float]] = None) -> None:
            try:
                super().interact(agents, global_context, nation_space, volatility_history, gdp_history, 
                                 market_data, policy)
                
                # Xử lý AgentPossessionLayer
                possession_layer = global_context.get("possession_layer", AgentPossessionLayer(self.nation))
                if hasattr(self, 'portfolio'):
                    possessed_asset = possession_layer.trigger_possession(self.id, self.portfolio)
                    if possessed_asset:
                        self.portfolio = {k: 0.0 for k in self.portfolio}
                        self.portfolio[possessed_asset] = 1.0
                        self.wealth *= 0.8
                        self.fear_index = min(1.0, self.fear_index + 0.4)
                        self.hope_index = min(1.0, self.hope_index + 0.3)
                
                # Xử lý ParallelEconomyLeak
                parallel_leak = global_context.get("parallel_leak", ParallelEconomyLeak(self.nation))
                outcome = parallel_leak.trigger_leak(self.id, self.wealth, market_data.get("Stock_Volatility", 0.0))
                if outcome != 0.0:
                    self.wealth = max(0, self.wealth + outcome)
                    if outcome < 0:
                        self.fear_index += 0.3
                        self.hope_index -= 0.2
                    else:
                        self.hope_index += 0.3
                        self.fear_index -= 0.1
            except Exception as e:
                logging.error(f"Error in interact for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ các tầng mới
def enhance_shadow_agent_for_combined_layers(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                bubble_risk = global_context.get("bubble_risk", 0.0)
                if bubble_risk > 0.6:
                    self.black_market_flow += self.wealth * 0.1
                    self.trust_government = max(0.0, self.trust_government - 0.1)
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

        def move_wealth_to_gold(self, gold_price: float):
            try:
                possession_layer = global_context.get("possession_layer", AgentPossessionLayer(self.nation))
                parallel_leak = global_context.get("parallel_leak", ParallelEconomyLeak(self.nation))
                
                if hasattr(self, 'portfolio'):
                    possessed_asset = possession_layer.trigger_possession(self.id, self.portfolio)
                    if possessed_asset:
                        self.portfolio = {k: 0.0 for k in self.portfolio}
                        self.portfolio[possessed_asset] = 1.0
                        self.wealth *= 0.8
                        self.gold_holdings = self.portfolio.get("gold", 0.0) * self.wealth / gold_price
                        self.cash_holdings = self.portfolio.get("cash", 0.0) * self.wealth
                        self.wealth = self.cash_holdings + self.gold_holdings * gold_price + \
                                     self.portfolio.get("crypto", 0.0) * self.wealth
                        self.stress_hormone = min(1.0, self.stress_hormone + 0.3) if hasattr(self, 'stress_hormone') else 0.5
                        self.activity_log.append({"action": "possessed", "asset": possessed_asset})
                    else:
                        outcome = parallel_leak.trigger_leak(self.id, self.wealth, global_context.get("Stock_Volatility", 0.0))
                        if outcome != 0.0:
                            self.wealth = max(0, self.wealth + outcome)
                            self.black_market_flow += abs(outcome) * 0.1
                            self.stress_hormone = min(1.0, self.stress_hormone + 0.2 if outcome < 0 else self.stress_hormone - 0.1) \
                                                 if hasattr(self, 'stress_hormone') else 0.5
                            self.activity_log.append({"action": "parallel_leak", "outcome": outcome})
                        else:
                            super().move_wealth_to_gold(gold_price)
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp các tầng mới vào VoTranhAbyssCoreMicro
def integrate_combined_layers(core, nation_name: str):
    try:
        core.echo_chamber = getattr(core, 'echo_chamber', {})
        core.possession_layer = getattr(core, 'possession_layer', {})
        core.parallel_leak = getattr(core, 'parallel_leak', {})
        
        core.echo_chamber[nation_name] = EchoChamberEffect(nation_name)
        core.possession_layer[nation_name] = AgentPossessionLayer(nation_name)
        core.parallel_leak[nation_name] = ParallelEconomyLeak(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_combined_layers(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_combined_layers(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
        
        # Xây dựng belief graph cho EchoChamberEffect
        agents = [a for a in core.agents if a.nation == nation_name]
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            agents += core.shadow_economies[nation_name].agents
        core.echo_chamber[nation_name].build_belief_graph(agents)
        
        logging.info(f"Integrated EchoChamberEffect, AgentPossessionLayer, and ParallelEconomyLeak for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_combined_layers for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm các tầng mới
def enhanced_reflect_economy_with_combined_layers(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                                 R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if nation_name in self.echo_chamber:
            echo_chamber = self.echo_chamber[nation_name]
            possession = self.possession_layer[nation_name]
            parallel_leak = self.parallel_leak[nation_name]
            
            self.global_context["possession_layer"] = possession
            self.global_context["parallel_leak"] = parallel_leak
            
            agents = [a for a in self.agents if a.nation == nation_name]
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                agents += self.shadow_economies[nation_name].agents
            
            context = {**self.global_context, **space}
            echo_chamber.update_beliefs(agents, context)
            possession.update_hysteria(space)
            
            echo_metrics = echo_chamber.get_metrics()
            possession_metrics = possession.get_metrics()
            leak_metrics = parallel_leak.get_metrics()
            
            self.global_context["bubble_risk"] = echo_metrics["bubble_risk"]
            
            # Tác động của EchoChamberEffect
            if echo_metrics["bubble_risk"] > 0.6:
                space["consumption"] *= 1.2
                space["market_sentiment"] += 0.3
                space["fear_index"] += 0.2
                space["resilience"] -= 0.1
                result["Insight"]["Psychology"] += f" | Belief bubble risk ({echo_metrics['bubble_risk']:.3f}) overheating market."
            
            # Tác động của AgentPossessionLayer
            if possession_metrics["mass_hysteria_level"] > 0.5:
                space["consumption"] *= 1.3
                space["market_sentiment"] += 0.3
                space["fear_index"] += 0.3
                space["resilience"] -= 0.2
                result["Insight"]["Psychology"] += f" | Mass hysteria ({possession_metrics['mass_hysteria_level']:.3f}) fueling erratic behavior."
            
            # Tác động của ParallelEconomyLeak
            if leak_metrics["leak_count"] > 10:
                space["consumption"] *= 1.1
                space["market_sentiment"] += 0.2
                space["resilience"] -= 0.1
                result["Insight"]["Psychology"] += f" | Parallel economy leaks ({leak_metrics['leak_count']}) amplifying volatility."
            
            # Ảnh hưởng đến shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_economy.cpi_impact += (echo_metrics["bubble_risk"] + possession_metrics["mass_hysteria_level"] + \
                                             leak_metrics["leaked_wealth"] / (shadow_economy.liquidity_pool + 1e-6)) * 0.1
                if echo_metrics["bubble_risk"] > 0.6 or possession_metrics["mass_hysteria_level"] > 0.5:
                    shadow_economy.liquidity_pool *= 1.15
                shadow_economy.liquidity_pool += leak_metrics["leaked_wealth"] * 0.2
            
            result["Echo_Chamber"] = echo_metrics
            result["Possession_Layer"] = possession_metrics
            result["Parallel_Leak"] = leak_metrics
            self.history[nation_name][-1]["echo_chamber_metrics"] = echo_metrics
            self.history[nation_name][-1]["possession_metrics"] = possession_metrics
            self.history[nation_name][-1]["parallel_leak_metrics"] = leak_metrics
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_combined_layers for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_combined_layers vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_combined_layers)

# Xuất dữ liệu các tầng
def export_combined_layers_data(core, nation_name: str):
    try:
        if hasattr(core, 'echo_chamber') and nation_name in core.echo_chamber:
            echo = core.echo_chamber[nation_name]
            data = {
                "Agent_ID": [n for n in echo.belief_graph.nodes()],
                "Trust_Government": [echo.belief_graph.nodes[n]["belief_vector"][0] for n in echo.belief_graph.nodes()]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"echo_chamber_{nation_name}.csv", index=False)
        
        if hasattr(core, 'possession_layer') and nation_name in core.possession_layer:
            possession = core.possession_layer[nation_name]
            data = {
                "Agent_ID": [h["agent_id"] for h in possession.possession_history],
                "Possessed_Asset": [h["asset"] for h in possession.possession_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"agent_possession_{nation_name}.csv", index=False)
        
        if hasattr(core, 'parallel_leak') and nation_name in core.parallel_leak:
            leak = core.parallel_leak[nation_name]
            data = {
                "Agent_ID": [h["agent_id"] for h in leak.leak_history],
                "Leak_Amount": [h["amount"] for h in leak.leak_history],
                "Outcome": [h["outcome"] for h in leak.leak_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"parallel_economy_leak_{nation_name}.csv", index=False)
        
        logging.info(f"Exported data for combined layers in {nation_name}")
    except Exception as e:
        logging.error(f"Error in export_combined_layers_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    integrate_investment_inertia(core, "Vietnam")
    integrate_mnemonic_market(core, "Vietnam")
    integrate_expectation_decay(core, "Vietnam")
    integrate_nostalgia_portfolio(core, "Vietnam")
    integrate_illusion_grid(core, "Vietnam")
    integrate_combined_layers(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_combined_layers_data(core, "Vietnam")
    print(f"Echo Chamber Metrics: {result.get('Echo_Chamber', {})}")
    print(f"Possession Layer Metrics: {result.get('Possession_Layer', {})}")
    print(f"Parallel Leak Metrics: {result.get('Parallel_Leak', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("entropy_archetype.log"), logging.StreamHandler()])

class EntropyBoundForecastDecay:
    def __init__(self, nation: str, entropy_threshold: float = 0.95, decay_rate: float = 0.03):
        self.nation = nation
        self.entropy_threshold = entropy_threshold
        self.decay_rate = decay_rate
        self.forecast_accuracy_history = deque(maxlen=7)  # Lưu lịch sử độ chính xác
        self.entropy_level = 0.0
        self.user_count = 0  # Đếm số lượng tác nhân sử dụng dự đoán

    def update_forecast(self, predicted_value: float, actual_value: float):
        """Cập nhật độ chính xác dự đoán và tính entropy."""
        try:
            accuracy = 1 - abs(predicted_value - actual_value) / (abs(actual_value) + 1e-6)
            self.forecast_accuracy_history.append(accuracy)
            self.user_count += 1
            
            if len(self.forecast_accuracy_history) >= 7:
                avg_accuracy = np.mean(self.forecast_accuracy_history)
                if avg_accuracy > self.entropy_threshold:
                    self.entropy_level = min(1.0, self.entropy_level + 0.1 * self.user_count / 1000)
                    decayed_prediction = predicted_value * (1 - self.entropy_level * random.gauss(0, self.decay_rate))
                    logging.debug(f"EntropyBoundForecastDecay {self.nation}: Decayed prediction to {decayed_prediction:.3f}, entropy {self.entropy_level:.3f}")
                    return decayed_prediction
            return predicted_value
        except Exception as e:
            logging.error(f"Error in update_forecast for {self.nation}: {e}")
            return predicted_value

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "entropy_level": self.entropy_level,
                "user_count": self.user_count,
                "avg_accuracy": np.mean(self.forecast_accuracy_history) if self.forecast_accuracy_history else 0.0
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

class ArchetypeEmergence:
    def __init__(self, nation: str, archetype_threshold: int = 1000):
        self.nation = nation
        self.archetype_threshold = archetype_threshold
        self.archetype_counts = {"Hoarder": 0, "Gambler": 0, "Prophet": 0, "Bureaucrat": 0}
        self.archetype_history = deque(maxlen=50)

    def assign_archetype(self, agent_id: str, behavior_sequence: List[Dict]):
        """Gán archetype dựa trên chuỗi hành vi."""
        try:
            if len(behavior_sequence) >= 5:
                wealth_changes = [b.get("wealth_change", 0.0) for b in behavior_sequence[-5:]]
                risk_levels = [b.get("risk_level", 0.0) for b in behavior_sequence[-5:]]
                
                if sum(1 for w in wealth_changes if w < 0) >= 4:  # Tiết kiệm liên tục
                    archetype = "Hoarder"
                elif max(risk_levels) > 0.8:  # Mạo hiểm cao
                    archetype = "Gambler"
                elif any(b.get("prediction_accuracy", 0.0) > 0.9 for b in behavior_sequence):  # Dự đoán chính xác
                    archetype = "Prophet"
                else:  # Chậm phản ứng
                    archetype = "Bureaucrat"
                
                self.archetype_counts[archetype] += 1
                self.archetype_history.append({"agent_id": agent_id, "archetype": archetype})
                logging.debug(f"ArchetypeEmergence {self.nation}: Assigned {archetype} to {agent_id}")
                return archetype
            return None
        except Exception as e:
            logging.error(f"Error in assign_archetype for {self.nation}: {e}")
            return None

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "hoarder_count": self.archetype_counts["Hoarder"],
                "gambler_count": self.archetype_counts["Gambler"],
                "prophet_count": self.archetype_counts["Prophet"],
                "bureaucrat_count": self.archetype_counts["Bureaucrat"]
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ EntropyBoundForecastDecay và ArchetypeEmergence
def enhance_hyper_agent_for_new_layers(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, 
                     trade_flow: float, resilience: float):
            super().__init__(id, nation, role, wealth, innovation, trade_flow, resilience)
            self.archetype = None
            self.behavior_sequence = deque(maxlen=5)

        def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                              volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                              market_momentum: float) -> None:
            try:
                super().update_psychology(global_context, nation_space, volatility_history, gdp_history, 
                                          sentiment, market_momentum)
                
                entropy_level = global_context.get("entropy_level", 0.0)
                if entropy_level > 0.5:
                    self.fear_index += 0.2
                    self.hope_index -= 0.1
                    if hasattr(self, 'risk_appetite'):
                        self.risk_appetite = max(0.0, self.risk_appetite - 0.1)
                
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "fear_index": self.fear_index,
                        "hope_index": self.hope_index,
                        "risk_appetite": getattr(self, 'risk_appetite', 0.5)
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.fear_index = adjusted_psych["fear_index"]
                    self.hope_index = adjusted_psych["hope_index"]
                    if hasattr(self, 'risk_appetite'):
                        self.risk_appetite = adjusted_psych["risk_appetite"]
            except Exception as e:
                logging.error(f"Error in update_psychology for {self.id}: {e}")

        def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                     volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                     policy: Optional[Dict[str, float]] = None) -> None:
            try:
                super().interact(agents, global_context, nation_space, volatility_history, gdp_history, 
                                 market_data, policy)
                
                # EntropyBoundForecastDecay
                entropy_layer = global_context.get("entropy_layer", EntropyBoundForecastDecay(self.nation))
                predicted_growth = global_context.get("predicted_growth", 0.0)
                actual_growth = market_data.get("market_momentum", 0.0) * 0.15
                decayed_prediction = entropy_layer.update_forecast(predicted_growth, actual_growth)
                global_context["predicted_growth"] = decayed_prediction
                
                # ArchetypeEmergence
                archetype_layer = global_context.get("archetype_layer", ArchetypeEmergence(self.nation))
                behavior = {
                    "wealth_change": (self.wealth - self.real_income_history[-2]) / (self.real_income_history[-2] + 1e-6) \
                                    if len(self.real_income_history) >= 2 else 0.0,
                    "risk_level": getattr(self, 'risk_appetite', 0.5),
                    "prediction_accuracy": 1 - abs(predicted_growth - actual_growth) / (abs(actual_growth) + 1e-6)
                }
                self.behavior_sequence.append(behavior)
                self.archetype = archetype_layer.assign_archetype(self.id, list(self.behavior_sequence))
                
                # Tác động của archetype
                if self.archetype == "Hoarder":
                    self.wealth *= 1.05
                    if hasattr(self, 'portfolio'):
                        self.portfolio["cash"] = min(1.0, self.portfolio.get("cash", 0.0) + 0.2)
                        total = sum(self.portfolio.values())
                        if total > 0:
                            self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                elif self.archetype == "Gambler" and hasattr(self, 'portfolio'):
                    self.portfolio["stocks"] = min(1.0, self.portfolio.get("stocks", 0.0) + 0.3)
                    total = sum(self.portfolio.values())
                    if total > 0:
                        self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                elif self.archetype == "Prophet":
                    self.hope_index += 0.2
                elif self.archetype == "Bureaucrat":
                    self.wealth *= 0.98
                
                logging.debug(f"HyperAgent {self.id}: Archetype {self.archetype}, Behavior {behavior}")
            except Exception as e:
                logging.error(f"Error in interact for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ EntropyBoundForecastDecay và ArchetypeEmergence
def enhance_shadow_agent_for_new_layers(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
            super().__init__(id, nation, wealth, trust_government)
            self.archetype = None
            self.behavior_sequence = deque(maxlen=5)

        def update_trust(self, inflation: float, government_stability: float, scandal_factor: float):
            try:
                super().update_trust(inflation, government_stability, scandal_factor)
                entropy_level = global_context.get("entropy_level", 0.0)
                if entropy_level > 0.5:
                    self.black_market_flow += self.wealth * 0.1
                    self.trust_government = max(0.0, self.trust_government - 0.1)
            except Exception as e:
                logging.error(f"Error in update_trust for {self.id}: {e}")

        def move_wealth_to_gold(self, gold_price: float):
            try:
                super().move_wealth_to_gold(gold_price)
                
                archetype_layer = global_context.get("archetype_layer", ArchetypeEmergence(self.nation))
                behavior = {
                    "wealth_change": (self.wealth - self.real_income_history[-2]) / (self.real_income_history[-2] + 1e-6) \
                                    if hasattr(self, 'real_income_history') and len(self.real_income_history) >= 2 else 0.0,
                    "risk_level": getattr(self, 'risk_appetite', 0.5),
                    "prediction_accuracy": 0.5
                }
                self.behavior_sequence.append(behavior)
                self.archetype = archetype_layer.assign_archetype(self.id, list(self.behavior_sequence))
                
                if self.archetype == "Hoarder" and hasattr(self, 'portfolio'):
                    self.portfolio["gold"] = min(1.0, self.portfolio.get("gold", 0.0) + 0.2)
                    total = sum(self.portfolio.values())
                    if total > 0:
                        self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                    self.gold_holdings = self.portfolio.get("gold", 0.0) * self.wealth / gold_price
                    self.cash_holdings = self.portfolio.get("cash", 0.0) * self.wealth
                    self.wealth = self.cash_holdings + self.gold_holdings * gold_price + \
                                 self.portfolio.get("crypto", 0.0) * self.wealth
                elif self.archetype == "Gambler":
                    self.black_market_flow += self.wealth * 0.1
                elif self.archetype == "Bureaucrat":
                    self.black_market_flow *= 0.95
                
                logging.debug(f"ShadowAgent {self.id}: Archetype {self.archetype}")
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp EntropyBoundForecastDecay và ArchetypeEmergence
def integrate_new_layers(core, nation_name: str):
    try:
        core.entropy_layer = getattr(core, 'entropy_layer', {})
        core.archetype_layer = getattr(core, 'archetype_layer', {})
        
        core.entropy_layer[nation_name] = EntropyBoundForecastDecay(nation_name)
        core.archetype_layer[nation_name] = ArchetypeEmergence(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_new_layers(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
            agent.archetype = None
            agent.behavior_sequence = deque(maxlen=5)
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_new_layers(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
                agent.archetype = None
                agent.behavior_sequence = deque(maxlen=5)
        
        logging.info(f"Integrated EntropyBoundForecastDecay and ArchetypeEmergence for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_new_layers for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm các tầng mới
def enhanced_reflect_economy_with_new_layers(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                            R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if nation_name in self.entropy_layer:
            entropy_layer = self.entropy_layer[nation_name]
            archetype_layer = self.archetype_layer[nation_name]
            
            self.global_context["entropy_layer"] = entropy_layer
            self.global_context["archetype_layer"] = archetype_layer
            
            entropy_metrics = entropy_layer.get_metrics()
            archetype_metrics = archetype_layer.get_metrics()
            
            # Tác động của EntropyBoundForecastDecay
            if entropy_metrics["entropy_level"] > 0.5:
                space["market_sentiment"] -= 0.2
                space["fear_index"] += 0.3
                space["resilience"] -= 0.1
                result["Insight"]["Psychology"] += f" | Forecast entropy ({entropy_metrics['entropy_level']:.3f}) disrupting predictions."
            
            # Tác động của ArchetypeEmergence
            total_archetypes = sum(archetype_metrics.values())
            if total_archetypes > 1000:
                space["consumption"] *= 1.1 if archetype_metrics["Gambler"] > archetype_metrics["Hoarder"] else 0.9
                space["market_sentiment"] += 0.2 if archetype_metrics["Prophet"] > 300 else -0.1
                result["Insight"]["Psychology"] += f" | Archetypes emerging: Gambler {archetype_metrics['gambler_count']}, "
                                                 f"Hoarder {archetype_metrics['hoarder_count']}."
            
            # Ảnh hưởng đến shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_economy.cpi_impact += entropy_metrics["entropy_level"] * 0.1
                if archetype_metrics["Gambler"] > 300:
                    shadow_economy.liquidity_pool *= 1.1
                shadow_economy.tax_loss += total_archetypes / 1000 * 0.05 * shadow_economy.liquidity_pool
            
            result["Entropy_Layer"] = entropy_metrics
            result["Archetype_Layer"] = archetype_metrics
            self.history[nation_name][-1]["entropy_metrics"] = entropy_metrics
            self.history[nation_name][-1]["archetype_metrics"] = archetype_metrics
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_new_layers for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_new_layers vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_new_layers)

# Xuất dữ liệu các tầng
def export_new_layers_data(core, nation_name: str):
    try:
        if hasattr(core, 'entropy_layer') and nation_name in core.entropy_layer:
            entropy = core.entropy_layer[nation_name]
            data = {
                "Step": list(range(len(entropy.forecast_accuracy_history))),
                "Forecast_Accuracy": list(entropy.forecast_accuracy_history),
                "Entropy_Level": [entropy.entropy_level] * len(entropy.forecast_accuracy_history)
            }
            df = pd.DataFrame(data)
            df.to_csv(f"entropy_decay_{nation_name}.csv", index=False)
        
        if hasattr(core, 'archetype_layer') and nation_name in core.archetype_layer:
            archetype = core.archetype_layer[nation_name]
            data = {
                "Agent_ID": [h["agent_id"] for h in archetype.archetype_history],
                "Archetype": [h["archetype"] for h in archetype.archetype_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"archetype_emergence_{nation_name}.csv", index=False)
        
        logging.info(f"Exported data for new layers in {nation_name}")
    except Exception as e:
        logging.error(f"Error in export_new_layers_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    integrate_investment_inertia(core, "Vietnam")
    integrate_mnemonic_market(core, "Vietnam")
    integrate_expectation_decay(core, "Vietnam")
    integrate_nostalgia_portfolio(core, "Vietnam")
    integrate_illusion_grid(core, "Vietnam")
    integrate_echo_chamber(core, "Vietnam")
    integrate_possession_layer(core, "Vietnam")
    integrate_parallel_leak(core, "Vietnam")
    integrate_new_layers(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_new_layers_data(core, "Vietnam")
    print(f"Entropy Layer Metrics: {result.get('Entropy_Layer', {})}")
    print(f"Archetype Layer Metrics: {result.get('Archetype_Layer', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("dream_meme_quantum.log"), logging.StreamHandler()])

class DreamStateMarketShifts:
    def __init__(self, nation: str, dream_prob: float = 0.05):
        self.nation = nation
        self.dream_prob = dream_prob
        self.dream_history = deque(maxlen=50)
        self.chaos_level = 0.0

    def trigger_dream_state(self, agent_id: str, portfolio: Dict[str, float]) -> bool:
        try:
            if random.random() < self.dream_prob:
                self.dream_history.append({"agent_id": agent_id})
                self.chaos_level = min(1.0, self.chaos_level + 0.1)
                logging.debug(f"DreamStateMarketShifts {self.nation}: Agent {agent_id} entered dream state")
                return True
            return False
        except Exception as e:
            logging.error(f"Error in trigger_dream_state for {self.nation}: {e}")
            return False

    def apply_dream_logic(self, portfolio: Dict[str, float], context: Dict[str, float]) -> Dict[str, float]:
        try:
            volatility = context.get("Stock_Volatility", 0.0)
            fear_index = context.get("fear_index", 0.0)
            value = math.sqrt(context.get("market_sentiment", 0.0)) / (fear_index + 1e-6)
            dream_portfolio = {k: value * random.uniform(0.5, 1.5) for k in portfolio}
            total = sum(dream_portfolio.values())
            dream_portfolio = {k: v / total for k, v in dream_portfolio.items()} if total > 0 else portfolio
            return dream_portfolio
        except Exception as e:
            logging.error(f"Error in apply_dream_logic for {self.nation}: {e}")
            return portfolio

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "chaos_level": self.chaos_level,
                "dream_count": len(self.dream_history)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

class InfectiousMemes:
    def __init__(self, nation: str, spread_rate: float = 0.1, toxicity: float = 0.3):
        self.nation = nation
        self.spread_rate = spread_rate
        self.toxicity = toxicity
        self.meme_history = deque(maxlen=50)
        self.infection_level = 0.0

    def spawn_meme(self, context: Dict[str, float]) -> Dict[str, float]:
        try:
            volatility = context.get("Stock_Volatility", 0.0)
            content = random.choice(["Sell all cash", "Buy uranium ETFs", "Crypto to the moon"])
            meme = {
                "content": content,
                "spread_rate": self.spread_rate * (1 + volatility),
                "toxicity": self.toxicity,
                "adoption": 0.0
            }
            self.meme_history.append(meme)
            logging.debug(f"InfectiousMemes {self.nation}: Spawned meme '{content}'")
            return meme
        except Exception as e:
            logging.error(f"Error in spawn_meme for {self.nation}: {e}")
            return {}

    def spread_meme(self, agents: List, meme: Dict[str, float]):
        try:
            for agent in agents:
                if random.random() < meme["spread_rate"]:
                    meme["adoption"] += 1.0 / len(agents)
                    if hasattr(agent, 'portfolio'):
                        if meme["content"] == "Sell all cash":
                            agent.portfolio["cash"] = max(0.0, agent.portfolio.get("cash", 0.0) - meme["toxicity"])
                        elif meme["content"] == "Buy uranium ETFs":
                            agent.portfolio["stocks"] = min(1.0, agent.portfolio.get("stocks", 0.0) + meme["toxicity"])
                        elif meme["content"] == "Crypto to the moon":
                            agent.portfolio["crypto"] = min(1.0, agent.portfolio.get("crypto", 0.0) + meme["toxicity"])
                        total = sum(agent.portfolio.values())
                        if total > 0:
                            agent.portfolio = {k: v / total for k, v in agent.portfolio.items()}
                    self.infection_level = min(1.0, self.infection_level + 0.01)
            logging.debug(f"InfectiousMemes {self.nation}: Meme adoption {meme['adoption']:.3f}")
        except Exception as e:
            logging.error(f"Error in spread_meme for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "infection_level": self.infection_level,
                "meme_count": len(self.meme_history),
                "avg_adoption": np.mean([m["adoption"] for m in self.meme_history]) if self.meme_history else 0.0
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

class QuantumDualityPortfolio:
    def __init__(self, nation: str, collapse_prob: float = 0.1):
        self.nation = nation
        self.collapse_prob = collapse_prob
        self.duality_history = deque(maxlen=50)
        self.shock_level = 0.0

    def create_superposition(self, portfolio: Dict[str, float]) -> List[Dict[str, float]]:
        try:
            safe = {k: v * 0.7 for k, v in portfolio.items()}
            safe["cash"] = safe.get("cash", 0.0) + 0.3
            total_safe = sum(safe.values())
            safe = {k: v / total_safe for k, v in safe.items()} if total_safe > 0 else safe
            
            risky = {k: v * 1.3 for k in ["stocks", "crypto"]}
            risky["cash"] = 0.1
            total_risky = sum(risky.values())
            risky = {k: v / total_risky for k, v in risky.items()} if total_risky > 0 else risky
            
            return [safe, risky]
        except Exception as e:
            logging.error(f"Error in create_superposition for {self.nation}: {e}")
            return [portfolio]

    def collapse_portfolio(self, agent_id: str, superpositions: List[Dict[str, float]]) -> Dict[str, float]:
        try:
            if random.random() < self.collapse_prob:
                chosen = random.choice(superpositions)
                self.duality_history.append({"agent_id": agent_id, "portfolio": chosen})
                self.shock_level = min(1.0, self.shock_level + 0.05)
                logging.debug(f"QuantumDualityPortfolio {self.nation}: Agent {agent_id} collapsed to {chosen}")
                return chosen
            return superpositions[0]  # Mặc định chọn safe
        except Exception as e:
            logging.error(f"Error in collapse_portfolio for {self.nation}: {e}")
            return superpositions[0]

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "shock_level": self.shock_level,
                "collapse_count": len(self.duality_history)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ các tầng mới
def enhance_hyper_agent_for_new_layers(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                     volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                     policy: Optional[Dict[str, float]] = None) -> None:
            try:
                super().interact(agents, global_context, nation_space, volatility_history, gdp_history, 
                                 market_data, policy)
                
                if hasattr(self, 'portfolio'):
                    # DreamStateMarketShifts
                    dream_layer = global_context.get("dream_layer", DreamStateMarketShifts(self.nation))
                    if dream_layer.trigger_dream_state(self.id, self.portfolio):
                        self.portfolio = dream_layer.apply_dream_logic(self.portfolio, global_context)
                        self.fear_index += 0.2
                        self.hope_index -= 0.1
                    
                    # InfectiousMemes
                    meme_layer = global_context.get("meme_layer", InfectiousMemes(self.nation))
                    active_meme = global_context.get("active_meme", None)
                    if active_meme and active_meme["adoption"] < 0.3:
                        meme_layer.spread_meme([self], active_meme)
                        if active_meme["adoption"] > 0.3:
                            global_context["meme_collapse"] = True
                            self.fear_index += 0.4
                            self.wealth *= 0.7
                    
                    # QuantumDualityPortfolio
                    quantum_layer = global_context.get("quantum_layer", QuantumDualityPortfolio(self.nation))
                    superpositions = quantum_layer.create_superposition(self.portfolio)
                    self.portfolio = quantum_layer.collapse_portfolio(self.id, superpositions)
                
                logging.debug(f"HyperAgent {self.id}: Updated with dream, meme, quantum layers")
            except Exception as e:
                logging.error(f"Error in interact for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ các tầng mới
def enhance_shadow_agent_for_new_layers(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def move_wealth_to_gold(self, gold_price: float):
            try:
                super().move_wealth_to_gold(gold_price)
                
                if hasattr(self, 'portfolio'):
                    dream_layer = global_context.get("dream_layer", DreamStateMarketShifts(self.nation))
                    meme_layer = global_context.get("meme_layer", InfectiousMemes(self.nation))
                    quantum_layer = global_context.get("quantum_layer", QuantumDualityPortfolio(self.nation))
                    
                    # DreamStateMarketShifts
                    if dream_layer.trigger_dream_state(self.id, self.portfolio):
                        self.portfolio = dream_layer.apply_dream_logic(self.portfolio, global_context)
                        self.stress_hormone = min(1.0, self.stress_hormone + 0.2) if hasattr(self, 'stress_hormone') else 0.5
                    
                    # InfectiousMemes
                    active_meme = global_context.get("active_meme", None)
                    if active_meme and active_meme["adoption"] < 0.3:
                        meme_layer.spread_meme([self], active_meme)
                        if active_meme["adoption"] > 0.3:
                            global_context["meme_collapse"] = True
                            self.black_market_flow += self.wealth * 0.2
                    
                    # QuantumDualityPortfolio
                    superpositions = quantum_layer.create_superposition(self.portfolio)
                    self.portfolio = quantum_layer.collapse_portfolio(self.id, superpositions)
                    
                    self.gold_holdings = self.portfolio.get("gold", 0.0) * self.wealth / gold_price
                    self.cash_holdings = self.portfolio.get("cash", 0.0) * self.wealth
                    self.wealth = self.cash_holdings + self.gold_holdings * gold_price + \
                                 self.portfolio.get("crypto", 0.0) * self.wealth
                    self.activity_log.append({"action": "portfolio_update", "portfolio": self.portfolio})
                
                logging.debug(f"ShadowAgent {self.id}: Updated with dream, meme, quantum layers")
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp các tầng mới vào VoTranhAbyssCoreMicro
def integrate_new_layers_21_23(core, nation_name: str):
    try:
        core.dream_layer = getattr(core, 'dream_layer', {})
        core.meme_layer = getattr(core, 'meme_layer', {})
        core.quantum_layer = getattr(core, 'quantum_layer', {})
        
        core.dream_layer[nation_name] = DreamStateMarketShifts(nation_name)
        core.meme_layer[nation_name] = InfectiousMemes(nation_name)
        core.quantum_layer[nation_name] = QuantumDualityPortfolio(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_new_layers(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_new_layers(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
        
        logging.info(f"Integrated DreamStateMarketShifts, InfectiousMemes, QuantumDualityPortfolio for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_new_layers_21_23 for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm các tầng mới
def enhanced_reflect_economy_with_new_layers(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                            R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if nation_name in self.dream_layer:
            dream_layer = self.dream_layer[nation_name]
            meme_layer = self.meme_layer[nation_name]
            quantum_layer = self.quantum_layer[nation_name]
            
            self.global_context["dream_layer"] = dream_layer
            self.global_context["meme_layer"] = meme_layer
            self.global_context["quantum_layer"] = quantum_layer
            
            # Spawn meme nếu cần
            if random.random() < 0.1 and "active_meme" not in self.global_context:
                self.global_context["active_meme"] = meme_layer.spawn_meme(space)
            
            dream_metrics = dream_layer.get_metrics()
            meme_metrics = meme_layer.get_metrics()
            quantum_metrics = quantum_layer.get_metrics()
            
            # Tác động của DreamStateMarketShifts
            if dream_metrics["chaos_level"] > 0.5:
                space["consumption"] *= 1.2
                space["market_sentiment"] += 0.2
                space["fear_index"] += 0.3
                result["Insight"]["Psychology"] += f" | Dream state chaos ({dream_metrics['chaos_level']:.3f}) warping market."
            
            # Tác động của InfectiousMemes
            if meme_metrics["infection_level"] > 0.5 or self.global_context.get("meme_collapse", False):
                space["consumption"] *= 0.7 if self.global_context.get("meme_collapse", False) else 1.3
                space["market_sentiment"] -= 0.3 if self.global_context.get("meme_collapse", False) else 0.2
                space["resilience"] -= 0.2
                result["Insight"]["Psychology"] += f" | Meme infection ({meme_metrics['infection_level']:.3f}) "
                                                 f"{'crash' if self.global_context.get('meme_collapse', False) else 'spreading'}."
                if self.global_context.get("meme_collapse", False):
                    self.global_context.pop("active_meme", None)
                    self.global_context.pop("meme_collapse", None)
            
            # Tác động của QuantumDualityPortfolio
            if quantum_metrics["shock_level"] > 0.5:
                space["market_sentiment"] -= 0.2
                space["fear_index"] += 0.3
                space["resilience"] -= 0.1
                result["Insight"]["Psychology"] += f" | Quantum collapse shock ({quantum_metrics['shock_level']:.3f}) destabilizing market."
            
            # Ảnh hưởng đến shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_economy.cpi_impact += (dream_metrics["chaos_level"] + meme_metrics["infection_level"] + quantum_metrics["shock_level"]) * 0.1
                if dream_metrics["chaos_level"] > 0.5 or meme_metrics["infection_level"] > 0.5:
                    shadow_economy.liquidity_pool *= 1.15
                shadow_economy.tax_loss += quantum_metrics["collapse_count"] / 100 * 0.05 * shadow_economy.liquidity_pool
            
            result["Dream_Layer"] = dream_metrics
            result["Meme_Layer"] = meme_metrics
            result["Quantum_Layer"] = quantum_metrics
            self.history[nation_name][-1]["dream_metrics"] = dream_metrics
            self.history[nation_name][-1]["meme_metrics"] = meme_metrics
            self.history[nation_name][-1]["quantum_metrics"] = quantum_metrics
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_new_layers for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_new_layers vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_new_layers)

# Xuất dữ liệu các tầng
def export_new_layers_data(core, nation_name: str):
    try:
        if hasattr(core, 'dream_layer') and nation_name in core.dream_layer:
            dream = core.dream_layer[nation_name]
            data = {
                "Agent_ID": [h["agent_id"] for h in dream.dream_history],
                "Dream_Step": [h.get("step", 0) for h in dream.dream_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"dream_state_{nation_name}.csv", index=False)
        
        if hasattr(core, 'meme_layer') and nation_name in core.meme_layer:
            meme = core.meme_layer[nation_name]
            data = {
                "Meme_Content": [h["content"] for h in meme.meme_history],
                "Adoption": [h["adoption"] for h in meme.meme_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"infectious_memes_{nation_name}.csv", index=False)
        
        if hasattr(core, 'quantum_layer') and nation_name in core.quantum_layer:
            quantum = core.quantum_layer[nation_name]
            data = {
                "Agent_ID": [h["agent_id"] for h in quantum.duality_history],
                "Collapsed_Stocks": [h["portfolio"].get("stocks", 0.0) for h in quantum.duality_history],
                "Collapsed_Cash": [h["portfolio"].get("cash", 0.0) for h in quantum.duality_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"quantum_duality_{nation_name}.csv", index=False)
        
        logging.info(f"Exported data for layers 21-23 in {nation_name}")
    except Exception as e:
        logging.error(f"Error in export_new_layers_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    integrate_investment_inertia(core, "Vietnam")
    integrate_mnemonic_market(core, "Vietnam")
    integrate_expectation_decay(core, "Vietnam")
    integrate_nostalgia_portfolio(core, "Vietnam")
    integrate_illusion_grid(core, "Vietnam")
    integrate_echo_chamber(core, "Vietnam")
    integrate_possession_layer(core, "Vietnam")
    integrate_parallel_leak(core, "Vietnam")
    integrate_new_layers_21_23(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_new_layers_data(core, "Vietnam")
    print(f"Dream Layer Metrics: {result.get('Dream_Layer', {})}")
    print(f"Meme Layer Metrics: {result.get('Meme_Layer', {})}")
    print(f"Quantum Layer Metrics: {result.get('Quantum_Layer', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("layers_24_28.log"), logging.StreamHandler()])

class NarrativeEngine:
    def __init__(self, nation: str, narrative_strength: float = 0.5):
        self.nation = nation
        self.narrative_strength = narrative_strength
        self.narrative_history = deque(maxlen=50)
        self.contrarian_count = 0

    def generate_narrative(self, context: Dict[str, float]) -> Dict[str, float]:
        try:
            volatility = context.get("Stock_Volatility", 0.0)
            title = random.choice([
                "Gold Is Dead" if context.get("market_sentiment", 0.0) < 0.0 else "Gold Revival Imminent",
                "Crypto Surge Ahead" if volatility > 0.5 else "Crypto Stability Returns",
                "Bonds Are Safe Haven" if context.get("fear_index", 0.0) > 0.5 else "Stocks To Soar"
            ])
            narrative = {
                "title": title,
                "impact": {
                    "gold": -0.2 if "Gold Is Dead" in title else 0.2,
                    "crypto": 0.3 if "Crypto" in title else 0.0,
                    "stocks": 0.2 if "Stocks" in title else 0.0,
                    "bonds": 0.2 if "Bonds" in title else 0.0
                },
                "strength": self.narrative_strength * (1 + volatility)
            }
            self.narrative_history.append(narrative)
            logging.debug(f"NarrativeEngine {self.nation}: Generated narrative '{title}'")
            return narrative
        except Exception as e:
            logging.error(f"Error in generate_narrative for {self.nation}: {e}")
            return {"title": "", "impact": {}, "strength": 0.0}

    def trigger_contrarian(self, agent_id: str):
        try:
            self.contrarian_count += 1
            logging.debug(f"NarrativeEngine {self.nation}: Agent {agent_id} became contrarian")
        except Exception as e:
            logging.error(f"Error in trigger_contrarian for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "narrative_strength": self.narrative_strength,
                "narrative_count": len(self.narrative_history),
                "contrarian_count": self.contrarian_count
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

class EconomicNecromancy:
    def __init__(self, nation: str, revival_prob: float = 0.01):
        self.nation = nation
        self.revival_prob = revival_prob
        self.necro_assets = ["Tulip_1637", "LUNA_2022", "BearStearns_2008"]
        self.necro_history = deque(maxlen=50)
        self.necro_trader_count = 0

    def revive_asset(self, agent_id: str, context: Dict[str, float]) -> Optional[str]:
        try:
            if random.random() < self.revival_prob:
                asset = random.choice(self.necro_assets)
                self.necro_history.append({"agent_id": agent_id, "asset": asset})
                logging.debug(f"EconomicNecromancy {self.nation}: Agent {agent_id} revived {asset}")
                return asset
            return None
        except Exception as e:
            logging.error(f"Error in revive_asset for {self.nation}: {e}")
            return None

    def evolve_to_necro_trader(self, agent_id: str):
        try:
            self.necro_trader_count += 1
            logging.debug(f"EconomicNecromancy {self.nation}: Agent {agent_id} evolved to NecroTrader")
        except Exception as e:
            logging.error(f"Error in evolve_to_necro_trader for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "necro_trader_count": self.necro_trader_count,
                "revival_count": len(self.necro_history)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

class MemeMarketMechanics:
    def __init__(self, nation: str, mmi_threshold: float = 0.85):
        self.nation = nation
        self.mmi_threshold = mmi_threshold
        self.mmi_history = deque(maxlen=50)
        self.fomo_level = 0.0

    def update_mmi(self, asset: str, context: Dict[str, float]) -> float:
        try:
            volatility = context.get("Stock_Volatility", 0.0)
            sentiment = context.get("market_sentiment", 0.0)
            mmi = random.uniform(0.5, 0.9) * (1 + volatility + sentiment)
            self.mmi_history.append({"asset": asset, "mmi": mmi})
            if mmi > self.mmi_threshold:
                self.fomo_level = min(1.0, self.fomo_level + 0.1)
            return mmi
        except Exception as e:
            logging.error(f"Error in update_mmi for {self.nation}: {e}")
            return 0.0

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "fomo_level": self.fomo_level,
                "mmi_count": len(self.mmi_history),
                "avg_mmi": np.mean([h["mmi"] for h in self.mmi_history]) if self.mmi_history else 0.0
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

class QuantumVolatilityReflections:
    def __init__(self, nation: str, meta_factor: float = 0.1):
        self.nation = nation
        self.meta_factor = meta_factor
        self.meta_volatility = 0.0
        self.reflection_history = deque(maxlen=50)

    def reflect_volatility(self, base_volatility: float, observer_count: int) -> float:
        try:
            meta_volatility = base_volatility * self.meta_factor * math.log(1 + observer_count)
            self.meta_volatility = min(1.0, meta_volatility)
            self.reflection_history.append({"base_volatility": base_volatility, "meta_volatility": self.meta_volatility})
            logging.debug(f"QuantumVolatilityReflections {self.nation}: Meta volatility {self.meta_volatility:.3f}")
            return self.meta_volatility
        except Exception as e:
            logging.error(f"Error in reflect_volatility for {self.nation}: {e}")
            return base_volatility

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "meta_volatility": self.meta_volatility,
                "reflection_count": len(self.reflection_history)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

class PsychopoliticalAgentFusion:
    def __init__(self, nation: str, fusion_factor: float = 0.5):
        self.nation = nation
        self.fusion_factor = fusion_factor
        self.fusion_history = deque(maxlen=50)
        self.hysteria_level = 0.0

    def fuse_agent(self, agent_id: str, economic_belief: float, political_emotion: float, ai_conflict: float):
        try:
            fused_belief = self.fusion_factor * economic_belief + (1 - self.fusion_factor) * (political_emotion + ai_conflict)
            self.fusion_history.append({"agent_id": agent_id, "fused_belief": fused_belief})
            self.hysteria_level = min(1.0, self.hysteria_level + 0.05 * abs(fused_belief))
            logging.debug(f"PsychopoliticalAgentFusion {self.nation}: Agent {agent_id} fused belief {fused_belief:.3f}")
            return fused_belief
        except Exception as e:
            logging.error(f"Error in fuse_agent for {self.nation}: {e}")
            return 0.0

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "hysteria_level": self.hysteria_level,
                "fusion_count": len(self.fusion_history)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ các tầng mới
def enhance_hyper_agent_for_new_layers(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, 
                     trade_flow: float, resilience: float):
            super().__init__(id, nation, role, wealth, innovation, trade_flow, resilience)
            self.fictional_belief = random.uniform(0.3, 0.7)
            self.is_contrarian = False
            self.is_necro_trader = False

        def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                     volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                     policy: Optional[Dict[str, float]] = None) -> None:
            try:
                super().interact(agents, global_context, nation_space, volatility_history, gdp_history, 
                                 market_data, policy)
                
                if hasattr(self, 'portfolio'):
                    # NarrativeEngine
                    narrative_layer = global_context.get("narrative_layer", NarrativeEngine(self.nation))
                    narrative = global_context.get("active_narrative", {"impact": {}, "strength": 0.0})
                    if narrative["strength"] > 0.7 and random.random() < 0.1 and not self.is_contrarian:
                        narrative_layer.trigger_contrarian(self.id)
                        self.is_contrarian = True
                    elif narrative["strength"] > 0.0 and not self.is_contrarian:
                        for asset, impact in narrative["impact"].items():
                            self.portfolio[asset] = max(0.0, min(1.0, self.portfolio.get(asset, 0.0) + impact))
                        total = sum(self.portfolio.values())
                        if total > 0:
                            self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                    
                    # EconomicNecromancy
                    necro_layer = global_context.get("necro_layer", EconomicNecromancy(self.nation))
                    necro_asset = necro_layer.revive_asset(self.id, global_context)
                    if necro_asset and not self.is_necro_trader:
                        self.portfolio[necro_asset] = 0.3
                        total = sum(self.portfolio.values())
                        if total > 0:
                            self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                        if random.random() < 0.05:
                            necro_layer.evolve_to_necro_trader(self.id)
                            self.is_necro_trader = True
                    
                    # MemeMarketMechanics
                    meme_layer = global_context.get("meme_market_layer", MemeMarketMechanics(self.nation))
                    for asset in self.portfolio:
                        mmi = meme_layer.update_mmi(asset, global_context)
                        if mmi > meme_layer.mmi_threshold:
                            self.portfolio[asset] = min(1.0, self.portfolio[asset] + 0.2)
                            total = sum(self.portfolio.values())
                            if total > 0:
                                self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                
                # PsychopoliticalAgentFusion
                fusion_layer = global_context.get("fusion_layer", PsychopoliticalAgentFusion(self.nation))
                self.fictional_belief = fusion_layer.fuse_agent(
                    self.id, 
                    self.trust_government, 
                    self.fear_index if hasattr(self, 'fear_index') else 0.0,
                    random.uniform(-0.5, 0.5)
                )
                if self.fictional_belief > 0.7:
                    self.wealth *= 1.1 if hasattr(self, 'portfolio') else 1.0
                elif self.fictional_belief < 0.3:
                    self.wealth *= 0.9
                
                logging.debug(f"HyperAgent {self.id}: Updated with narrative, necro, meme, quantum, fusion layers")
            except Exception as e:
                logging.error(f"Error in interact for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ các tầng mới
def enhance_shadow_agent_for_new_layers(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def __init__(self, id: str, nation: str, wealth: float, trust_government: float = 0.5):
            super().__init__(id, nation, wealth, trust_government)
            self.fictional_belief = random.uniform(0.2, 0.5)
            self.is_contrarian = False
            self.is_necro_trader = False

        def move_wealth_to_gold(self, gold_price: float):
            try:
                super().move_wealth_to_gold(gold_price)
                
                if hasattr(self, 'portfolio'):
                    narrative_layer = global_context.get("narrative_layer", NarrativeEngine(self.nation))
                    necro_layer = global_context.get("necro_layer", EconomicNecromancy(self.nation))
                    meme_layer = global_context.get("meme_market_layer", MemeMarketMechanics(self.nation))
                    fusion_layer = global_context.get("fusion_layer", PsychopoliticalAgentFusion(self.nation))
                    
                    # NarrativeEngine
                    narrative = global_context.get("active_narrative", {"impact": {}, "strength": 0.0})
                    if narrative["strength"] > 0.7 and random.random() < 0.05 and not self.is_contrarian:
                        narrative_layer.trigger_contrarian(self.id)
                        self.is_contrarian = True
                    elif narrative["strength"] > 0.0 and not self.is_contrarian:
                        for asset, impact in narrative["impact"].items():
                            self.portfolio[asset] = max(0.0, min(1.0, self.portfolio.get(asset, 0.0) + impact * 0.5))
                        total = sum(self.portfolio.values())
                        if total > 0:
                            self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                    
                    # EconomicNecromancy
                    necro_asset = necro_layer.revive_asset(self.id, global_context)
                    if necro_asset and not self.is_necro_trader:
                        self.portfolio[necro_asset] = 0.2
                        total = sum(self.portfolio.values())
                        if total > 0:
                            self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                        if random.random() < 0.03:
                            necro_layer.evolve_to_necro_trader(self.id)
                            self.is_necro_trader = True
                    
                    # MemeMarketMechanics
                    for asset in self.portfolio:
                        mmi = meme_layer.update_mmi(asset, global_context)
                        if mmi > meme_layer.mmi_threshold:
                            self.portfolio[asset] = min(1.0, self.portfolio[asset] + 0.15)
                            total = sum(self.portfolio.values())
                            if total > 0:
                                self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                    
                    self.gold_holdings = self.portfolio.get("gold", 0.0) * self.wealth / gold_price
                    self.cash_holdings = self.portfolio.get("cash", 0.0) * self.wealth
                    self.wealth = self.cash_holdings + self.gold_holdings * gold_price + \
                                 self.portfolio.get("crypto", 0.0) * self.wealth
                    self.activity_log.append({"action": "portfolio_update", "portfolio": self.portfolio})
                
                # PsychopoliticalAgentFusion
                self.fictional_belief = fusion_layer.fuse_agent(
                    self.id, 
                    self.trust_government, 
                    self.stress_hormone if hasattr(self, 'stress_hormone') else 0.0,
                    random.uniform(-0.3, 0.3)
                )
                if self.fictional_belief > 0.6:
                    self.black_market_flow += self.wealth * 0.1
                
                logging.debug(f"ShadowAgent {self.id}: Updated with narrative, necro, meme, quantum, fusion layers")
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp các tầng mới vào VoTranhAbyssCoreMicro
def integrate_new_layers_24_28(core, nation_name: str):
    try:
        core.narrative_layer = getattr(core, 'narrative_layer', {})
        core.necro_layer = getattr(core, 'necro_layer', {})
        core.meme_market_layer = getattr(core, 'meme_market_layer', {})
        core.quantum_volatility_layer = getattr(core, 'quantum_volatility_layer', {})
        core.fusion_layer = getattr(core, 'fusion_layer', {})
        
        core.narrative_layer[nation_name] = NarrativeEngine(nation_name)
        core.necro_layer[nation_name] = EconomicNecromancy(nation_name)
        core.meme_market_layer[nation_name] = MemeMarketMechanics(nation_name)
        core.quantum_volatility_layer[nation_name] = QuantumVolatilityReflections(nation_name)
        core.fusion_layer[nation_name] = PsychopoliticalAgentFusion(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_new_layers(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
            agent.fictional_belief = random.uniform(0.3, 0.7)
            agent.is_contrarian = False
            agent.is_necro_trader = False
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_new_layers(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
                agent.fictional_belief = random.uniform(0.2, 0.5)
                agent.is_contrarian = False
                agent.is_necro_trader = False
        
        logging.info(f"Integrated layers 24-28 for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_new_layers_24_28 for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm các tầng mới
def enhanced_reflect_economy_with_new_layers(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                            R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if nation_name in self.narrative_layer:
            narrative_layer = self.narrative_layer[nation_name]
            necro_layer = self.necro_layer[nation_name]
            meme_market_layer = self.meme_market_layer[nation_name]
            quantum_volatility_layer = self.quantum_volatility_layer[nation_name]
            fusion_layer = self.fusion_layer[nation_name]
            
            self.global_context["narrative_layer"] = narrative_layer
            self.global_context["necro_layer"] = necro_layer
            self.global_context["meme_market_layer"] = meme_market_layer
            self.global_context["fusion_layer"] = fusion_layer
            
            # Generate narrative nếu cần
            if random.random() < 0.2:
                self.global_context["active_narrative"] = narrative_layer.generate_narrative(space)
            
            # QuantumVolatilityReflections
            base_volatility = space.get("Stock_Volatility", 0.0)
            observer_count = len(self.agents) if hasattr(self, 'agents') else 1000
            space["Stock_Volatility"] = quantum_volatility_layer.reflect_volatility(base_volatility, observer_count)
            
            narrative_metrics = narrative_layer.get_metrics()
            necro_metrics = necro_layer.get_metrics()
            meme_metrics = meme_market_layer.get_metrics()
            quantum_metrics = quantum_volatility_layer.get_metrics()
            fusion_metrics = fusion_layer.get_metrics()
            
            # Tác động của NarrativeEngine
            if narrative_metrics["contrarian_count"] > 100:
                space["consumption"] *= 0.9
                space["market_sentiment"] -= 0.2
                result["Insight"]["Psychology"] += f" | Contrarians ({narrative_metrics['contrarian_count']}) resisting narrative."
            
            # Tác động của EconomicNecromancy
            if necro_metrics["necro_trader_count"] > 50:
                space["market_sentiment"] += 0.2
                space["fear_index"] += 0.1
                result["Insight"]["Psychology"] += f" | NecroTraders ({necro_metrics['necro_trader_count']}) reviving old assets."
            
            # Tác động của MemeMarketMechanics
            if meme_metrics["fomo_level"] > 0.5:
                space["consumption"] *= 1.2
                space["market_sentiment"] += 0.3
                space["resilience"] -= 0.1
                result["Insight"]["Psychology"] += f" | Meme FOMO ({meme_metrics['fomo_level']:.3f}) driving irrational trades."
            
            # Tác động của QuantumVolatilityReflections
            if quantum_metrics["meta_volatility"] > 0.5:
                space["fear_index"] += 0.2
                space["resilience"] -= 0.1
                result["Insight"]["Psychology"] += f" | Meta volatility ({quantum_metrics['meta_volatility']:.3f}) amplifying uncertainty."
            
            # Tác động của PsychopoliticalAgentFusion
            if fusion_metrics["hysteria_level"] > 0.5:
                space["consumption"] *= 1.3
                space["market_sentiment"] -= 0.2
                space["fear_index"] += 0.3
                result["Insight"]["Psychology"] += f" | Psychopolitical hysteria ({fusion_metrics['hysteria_level']:.3f}) warping decisions."
            
            # Ảnh hưởng đến shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_economy.cpi_impact += (narrative_metrics["narrative_strength"] + 
                                             meme_metrics["fomo_level"] + 
                                             fusion_metrics["hysteria_level"]) * 0.1
                shadow_economy.liquidity_pool *= 1.1 if necro_metrics["necro_trader_count"] > 50 else 1.0
                shadow_economy.tax_loss += quantum_metrics["meta_volatility"] * 0.05 * shadow_economy.liquidity_pool
            
            result["Narrative_Layer"] = narrative_metrics
            result["Necro_Layer"] = necro_metrics
            result["Meme_Market_Layer"] = meme_metrics
            result["Quantum_Volatility_Layer"] = quantum_metrics
            result["Fusion_Layer"] = fusion_metrics
            self.history[nation_name][-1]["narrative_metrics"] = narrative_metrics
            self.history[nation_name][-1]["necro_metrics"] = necro_metrics
            self.history[nation_name][-1]["meme_market_metrics"] = meme_metrics
            self.history[nation_name][-1]["quantum_volatility_metrics"] = quantum_metrics
            self.history[nation_name][-1]["fusion_metrics"] = fusion_metrics
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_new_layers for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_new_layers vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_new_layers)

# Xuất dữ liệu các tầng
def export_new_layers_data(core, nation_name: str):
    try:
        if hasattr(core, 'narrative_layer') and nation_name in core.narrative_layer:
            narrative = core.narrative_layer[nation_name]
            data = {
                "Narrative_Title": [h["title"] for h in narrative.narrative_history],
                "Strength": [h["strength"] for h in narrative.narrative_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"narrative_engine_{nation_name}.csv", index=False)
        
        if hasattr(core, 'necro_layer') and nation_name in core.necro_layer:
            necro = core.necro_layer[nation_name]
            data = {
                "Agent_ID": [h["agent_id"] for h in necro.necro_history],
                "Revived_Asset": [h["asset"] for h in necro.necro_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"economic_necromancy_{nation_name}.csv", index=False)
        
        if hasattr(core, 'meme_market_layer') and nation_name in core.meme_market_layer:
            meme = core.meme_market_layer[nation_name]
            data = {
                "Asset": [h["asset"] for h in meme.mmi_history],
                "MMI": [h["mmi"] for h in meme.mmi_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"meme_market_{nation_name}.csv", index=False)
        
        if hasattr(core, 'quantum_volatility_layer') and nation_name in core.quantum_volatility_layer:
            quantum = core.quantum_volatility_layer[nation_name]
            data = {
                "Base_Volatility": [h["base_volatility"] for h in quantum.reflection_history],
                "Meta_Volatility": [h["meta_volatility"] for h in quantum.reflection_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"quantum_volatility_{nation_name}.csv", index=False)
        
        if hasattr(core, 'fusion_layer') and nation_name in core.fusion_layer:
            fusion = core.fusion_layer[nation_name]
            data = {
                "Agent_ID": [h["agent_id"] for h in fusion.fusion_history],
                "Fused_Belief": [h["fused_belief"] for h in fusion.fusion_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"psychopolitical_fusion_{nation_name}.csv", index=False)
        
        logging.info(f"Exported data for layers 24-28 in {nation_name}")
    except Exception as e:
        logging.error(f"Error in export_new_layers_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    integrate_investment_inertia(core, "Vietnam")
    integrate_mnemonic_market(core, "Vietnam")
    integrate_expectation_decay(core, "Vietnam")
    integrate_nostalgia_portfolio(core, "Vietnam")
    integrate_illusion_grid(core, "Vietnam")
    integrate_echo_chamber(core, "Vietnam")
    integrate_possession_layer(core, "Vietnam")
    integrate_parallel_leak(core, "Vietnam")
    integrate_new_layers_24_28(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_new_layers_data(core, "Vietnam")
    print(f"Narrative Layer Metrics: {result.get('Narrative_Layer', {})}")
    print(f"Necro Layer Metrics: {result.get('Necro_Layer', {})}")
    print(f"Meme Market Layer Metrics: {result.get('Meme_Market_Layer', {})}")
    print(f"Quantum Volatility Layer Metrics: {result.get('Quantum_Volatility_Layer', {})}")
    print(f"Fusion Layer Metrics: {result.get('Fusion_Layer', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import logging
import pandas as pd
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("systemic_archetype_amplifier.log"), logging.StreamHandler()])

class SystemicArchetypeAmplifier:
    def __init__(self, nation: str, dominance_threshold: float = 0.4, amplification_factor: float = 1.5):
        self.nation = nation
        self.dominance_threshold = dominance_threshold  # Ngưỡng để một archetype thống trị
        self.amplification_factor = amplification_factor  # Hệ số khuếch đại hành vi
        self.archetype_distribution = {"Hoarder": 0, "Gambler": 0, "Prophet": 0, "Bureaucrat": 0}
        self.amplification_history = deque(maxlen=50)  # Lưu lịch sử khuếch đại
        self.wave_magnitude = 0.0  # Độ lớn của làn sóng kinh tế

    def update_distribution(self, agents: List):
        """Cập nhật phân phối archetype dựa trên trạng thái tác nhân."""
        try:
            total_agents = len(agents)
            if total_agents == 0:
                return
            
            # Đếm số lượng mỗi archetype
            self.archetype_distribution = {"Hoarder": 0, "Gambler": 0, "Prophet": 0, "Bureaucrat": 0}
            for agent in agents:
                archetype = getattr(agent, 'archetype', None)
                if archetype in self.archetype_distribution:
                    self.archetype_distribution[archetype] += 1
            
            # Chuẩn hóa thành tỷ lệ
            for archetype in self.archetype_distribution:
                self.archetype_distribution[archetype] /= total_agents
            
            logging.debug(f"SystemicArchetypeAmplifier {self.nation}: Archetype distribution {self.archetype_distribution}")
        except Exception as e:
            logging.error(f"Error in update_distribution for {self.nation}: {e}")

    def amplify_archetype(self, agents: List, context: Dict[str, float]) -> Optional[str]:
        """Khuếch đại hành vi nếu một archetype thống trị."""
        try:
            dominant_archetype = None
            for archetype, proportion in self.archetype_distribution.items():
                if proportion > self.dominance_threshold:
                    dominant_archetype = archetype
                    break
            
            if dominant_archetype:
                for agent in agents:
                    if getattr(agent, 'archetype', None) == dominant_archetype:
                        self._apply_amplification(agent, dominant_archetype, context)
                
                self.wave_magnitude = min(1.0, self.wave_magnitude + 0.1 * self.archetype_distribution[dominant_archetype])
                self.amplification_history.append({"archetype": dominant_archetype, "magnitude": self.wave_magnitude})
                logging.warning(f"SystemicArchetypeAmplifier {self.nation}: {dominant_archetype} dominates, wave magnitude {self.wave_magnitude:.3f}")
                return dominant_archetype
            else:
                self.wave_magnitude = max(0.0, self.wave_magnitude - 0.05)
                return None
        except Exception as e:
            logging.error(f"Error in amplify_archetype for {self.nation}: {e}")
            return None

    def _apply_amplification(self, agent, archetype: str, context: Dict[str, float]):
        """Áp dụng khuếch đại hành vi cho tác nhân thuộc archetype thống trị."""
        try:
            if archetype == "Hoarder":
                agent.wealth *= self.amplification_factor * 0.8
                if hasattr(agent, 'portfolio'):
                    agent.portfolio["cash"] = min(1.0, agent.portfolio.get("cash", 0.0) + 0.3 * self.amplification_factor)
                    total = sum(agent.portfolio.values())
                    if total > 0:
                        agent.portfolio = {k: v / total for k, v in agent.portfolio.items()}
                agent.fear_index = min(1.0, agent.fear_index + 0.2) if hasattr(agent, 'fear_index') else 0.5
            
            elif archetype == "Gambler":
                if hasattr(agent, 'portfolio'):
                    agent.portfolio["stocks"] = min(1.0, agent.portfolio.get("stocks", 0.0) + 0.4 * self.amplification_factor)
                    total = sum(agent.portfolio.values())
                    if total > 0:
                        agent.portfolio = {k: v / total for k, v in agent.portfolio.items()}
                agent.wealth *= random.uniform(0.7, 1.3) * self.amplification_factor
                agent.risk_appetite = min(1.0, agent.risk_appetite + 0.3) if hasattr(agent, 'risk_appetite') else 0.5
            
            elif archetype == "Prophet":
                agent.hope_index = min(1.0, agent.hope_index + 0.3 * self.amplification_factor) if hasattr(agent, 'hope_index') else 0.5
                agent.trust_government = max(0.0, agent.trust_government - 0.2) if context.get("market_sentiment", 0.0) < 0.0 else agent.trust_government
            
            elif archetype == "Bureaucrat":
                agent.wealth *= 0.95 * self.amplification_factor
                agent.consumption_state = "low" if hasattr(agent, 'consumption_state') else "normal"
                agent.fear_index = min(1.0, agent.fear_index + 0.1) if hasattr(agent, 'fear_index') else 0.5
            
            logging.debug(f"SystemicArchetypeAmplifier {self.nation}: Amplified {archetype} behavior for agent {agent.id}")
        except Exception as e:
            logging.error(f"Error in _apply_amplification for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Trả về các chỉ số của SystemicArchetypeAmplifier."""
        try:
            return {
                "wave_magnitude": self.wave_magnitude,
                "hoarder_proportion": self.archetype_distribution["Hoarder"],
                "gambler_proportion": self.archetype_distribution["Gambler"],
                "prophet_proportion": self.archetype_distribution["Prophet"],
                "bureaucrat_proportion": self.archetype_distribution["Bureaucrat"],
                "amplification_events": len(self.amplification_history)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ SystemicArchetypeAmplifier
def enhance_hyper_agent_for_archetype_amplifier(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                     volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                     policy: Optional[Dict[str, float]] = None) -> None:
            try:
                super().interact(agents, global_context, nation_space, volatility_history, gdp_history, 
                                 market_data, policy)
                
                archetype_amplifier = global_context.get("archetype_amplifier", SystemicArchetypeAmplifier(self.nation))
                dominant_archetype = global_context.get("dominant_archetype", None)
                
                if dominant_archetype and dominant_archetype == self.archetype:
                    # Tác động khuếch đại đã được áp dụng trong amplify_archetype
                    self.fear_index = min(1.0, self.fear_index + 0.1) if hasattr(self, 'fear_index') else 0.5
                    logging.debug(f"HyperAgent {self.id}: Affected by {dominant_archetype} amplification")
                
            except Exception as e:
                logging.error(f"Error in interact for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ SystemicArchetypeAmplifier
def enhance_shadow_agent_for_archetype_amplifier(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def move_wealth_to_gold(self, gold_price: float):
            try:
                super().move_wealth_to_gold(gold_price)
                
                archetype_amplifier = global_context.get("archetype_amplifier", SystemicArchetypeAmplifier(self.nation))
                dominant_archetype = global_context.get("dominant_archetype", None)
                
                if dominant_archetype and dominant_archetype == self.archetype:
                    if dominant_archetype == "Hoarder" and hasattr(self, 'portfolio'):
                        self.portfolio["gold"] = min(1.0, self.portfolio.get("gold", 0.0) + 0.2)
                        total = sum(self.portfolio.values())
                        if total > 0:
                            self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                        self.gold_holdings = self.portfolio.get("gold", 0.0) * self.wealth / gold_price
                        self.cash_holdings = self.portfolio.get("cash", 0.0) * self.wealth
                        self.wealth = self.cash_holdings + self.gold_holdings * gold_price + \
                                     self.portfolio.get("crypto", 0.0) * self.wealth
                        self.activity_log.append({"action": "amplified_hoarder", "portfolio": self.portfolio})
                    elif dominant_archetype == "Gambler":
                        self.black_market_flow += self.wealth * 0.15
                    logging.debug(f"ShadowAgent {self.id}: Affected by {dominant_archetype} amplification")
                
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp SystemicArchetypeAmplifier vào VoTranhAbyssCoreMicro
def integrate_archetype_amplifier(core, nation_name: str):
    """Tích hợp SystemicArchetypeAmplifier vào hệ thống chính."""
    try:
        core.archetype_amplifier = getattr(core, 'archetype_amplifier', {})
        core.archetype_amplifier[nation_name] = SystemicArchetypeAmplifier(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_archetype_amplifier(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_archetype_amplifier(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
        
        logging.info(f"Integrated SystemicArchetypeAmplifier for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_archetype_amplifier for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm SystemicArchetypeAmplifier
def enhanced_reflect_economy_with_archetype_amplifier(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                                     R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if hasattr(self, 'archetype_amplifier') and nation_name in self.archetype_amplifier:
            archetype_amplifier = self.archetype_amplifier[nation_name]
            self.global_context["archetype_amplifier"] = archetype_amplifier
            
            agents = [a for a in self.agents if a.nation == nation_name]
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                agents += self.shadow_economies[nation_name].agents
            
            # Cập nhật phân phối archetype
            archetype_amplifier.update_distribution(agents)
            
            # Kích hoạt khuếch đại nếu có archetype thống trị
            context = {**self.global_context, **space}
            dominant_archetype = archetype_amplifier.amplify_archetype(agents, context)
            self.global_context["dominant_archetype"] = dominant_archetype
            
            metrics = archetype_amplifier.get_metrics()
            
            # Tác động của khuếch đại lên hệ thống
            if metrics["wave_magnitude"] > 0.5:
                if dominant_archetype == "Hoarder":
                    space["consumption"] *= 0.7
                    space["fear_index"] += 0.3
                    space["resilience"] += 0.1
                elif dominant_archetype == "Gambler":
                    space["consumption"] *= 1.3
                    space["market_sentiment"] += 0.4
                    space["resilience"] -= 0.2
                elif dominant_archetype == "Prophet":
                    space["market_sentiment"] += 0.3
                    space["hope_index"] += 0.3
                elif dominant_archetype == "Bureaucrat":
                    space["consumption"] *= 0.8
                    space["resilience"] -= 0.1
                result["Insight"]["Psychology"] += f" | {dominant_archetype} wave (magnitude {metrics['wave_magnitude']:.3f}) reshaping market."
            
            # Ảnh hưởng đến shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                shadow_economy = self.shadow_economies[nation_name]
                shadow_economy.cpi_impact += metrics["wave_magnitude"] * 0.1
                if metrics["wave_magnitude"] > 0.5:
                    shadow_economy.liquidity_pool *= 1.1 if dominant_archetype == "Gambler" else 0.9
                shadow_economy.tax_loss += metrics["amplification_events"] / 100 * 0.05 * shadow_economy.liquidity_pool
            
            result["Archetype_Amplifier"] = metrics
            self.history[nation_name][-1]["archetype_amplifier_metrics"] = metrics
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_archetype_amplifier for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_archetype_amplifier vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_archetype_amplifier)

# Xuất dữ liệu SystemicArchetypeAmplifier
def export_archetype_amplifier_data(core, nation_name: str, filename: str = "archetype_amplifier_data.csv"):
    """Xuất dữ liệu SystemicArchetypeAmplifier."""
    try:
        if hasattr(core, 'archetype_amplifier') and nation_name in core.archetype_amplifier:
            amplifier = core.archetype_amplifier[nation_name]
            data = {
                "Step": list(range(len(amplifier.amplification_history))),
                "Dominant_Archetype": [h["archetype"] for h in amplifier.amplification_history],
                "Wave_Magnitude": [h["magnitude"] for h in amplifier.amplification_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"SystemicArchetypeAmplifier {nation_name}: Exported data to {filename}")
    except Exception as e:
        logging.error(f"Error in export_archetype_amplifier_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    # Giả lập tích hợp các tầng trước (1-27, 29-31)
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    integrate_investment_inertia(core, "Vietnam")
    integrate_mnemonic_market(core, "Vietnam")
    integrate_expectation_decay(core, "Vietnam")
    integrate_nostalgia_portfolio(core, "Vietnam")
    integrate_illusion_grid(core, "Vietnam")
    integrate_echo_chamber(core, "Vietnam")
    integrate_possession_layer(core, "Vietnam")
    integrate_parallel_leak(core, "Vietnam")
    integrate_new_layers_21_23(core, "Vietnam")
    integrate_new_layers_24_28(core, "Vietnam")
    integrate_singularity_layers(core, "Vietnam")
    
    # Tích hợp tầng 28
    integrate_archetype_amplifier(core, "Vietnam")
    
    # Mô phỏng một bước
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    # Xuất dữ liệu
    export_archetype_amplifier_data(core, "Vietnam", "archetype_amplifier_vietnam.csv")
    print(f"Archetype Amplifier Metrics: {result.get('Archetype_Amplifier', {})}")
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import logging
import pandas as pd
from collections import deque
import networkx as nx
from scipy.stats import entropy
from dataclasses import dataclass
import cupy as cp
import uuid

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("singularity_layers.log"), logging.StreamHandler()])

# Tầng 29: Endogenous Collapse Seeds
@dataclass
class CollapseSeed:
    id: str
    trigger_threshold: float
    impact_factor: float
    latency: int
    planted_step: float
    activated: bool = False

class EndogenousCollapseSeeds:
    def __init__(self, nation: str, seed_interval: int = 500, max_seeds: int = 50):
        self.nation = nation
        self.seed_interval = seed_interval
        self.max_seeds = max_seeds
        self.seeds = deque(maxlen=max_seeds)
        self.collapse_risk = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def plant_seed(self, step: float, context: Dict[str, float]):
        try:
            if step % self.seed_interval == 0 and len(self.seeds) < self.max_seeds:
                seed = CollapseSeed(
                    id=str(uuid.uuid4()),
                    trigger_threshold=random.uniform(0.7, 0.9),
                    impact_factor=random.uniform(0.5, 1.5),
                    latency=random.randint(100, 500),
                    planted_step=step
                )
                self.seeds.append(seed)
                logging.debug(f"EndogenousCollapseSeeds {self.nation}: Planted seed {seed.id} at step {step}")
        except Exception as e:
            logging.error(f"Error in plant_seed for {self.nation}: {e}")

    def check_seeds(self, step: float, context: Dict[str, float]) -> List[CollapseSeed]:
        try:
            activated_seeds = []
            pmi = context.get("pmi", 0.5)
            fear_index = context.get("fear_index", 0.0)
            stability_score = 1 - (pmi * 0.5 + (1 - fear_index) * 0.5)  # Thấp khi mọi thứ "tốt đẹp"
            
            for seed in self.seeds:
                if not seed.activated and step >= seed.planted_step + seed.latency and stability_score < seed.trigger_threshold:
                    seed.activated = True
                    activated_seeds.append(seed)
                    self.collapse_risk = min(1.0, self.collapse_risk + seed.impact_factor * 0.2)
                    logging.warning(f"EndogenousCollapseSeeds {self.nation}: Seed {seed.id} activated, risk {self.collapse_risk:.3f}")
            
            return activated_seeds
        except Exception as e:
            logging.error(f"Error in check_seeds for {self.nation}: {e}")
            return []

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "collapse_risk": self.collapse_risk,
                "active_seeds": sum(1 for s in self.seeds if s.activated),
                "total_seeds": len(self.seeds)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Tầng 30: Reflexive Necroeconomics
class ZombieEconomyNet(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)  # Output: [GDP, employment, trade]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        try:
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return torch.sigmoid(x)
        except Exception as e:
            logging.error(f"Error in ZombieEconomyNet forward: {e}")
            return x

class ReflexiveNecroeconomics:
    def __init__(self, nation: str, zombie_threshold: float = 0.1):
        self.nation = nation
        self.zombie_threshold = zombie_threshold
        self.zombie_net = ZombieEconomyNet().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer = torch.optim.Adam(self.zombie_net.parameters(), lr=0.0001)
        self.zombie_history = deque(maxlen=50)
        self.zombie_state = False
        self.reflexive_factor = 0.0

    def simulate_zombie_economy(self, context: Dict[str, float], agents: List) -> Dict[str, float]:
        try:
            consumption = sum(getattr(a, 'consumption', 0.0) for a in agents)
            if consumption < self.zombie_threshold * len(agents):
                self.zombie_state = True
                input_data = torch.tensor([
                    context.get("pmi", 0.5),
                    context.get("market_sentiment", 0.0),
                    context.get("fear_index", 0.0),
                    context.get("Stock_Volatility", 0.0),
                    context.get("inflation", 0.04),
                    context.get("trade", 1.0),
                    context.get("resilience", 0.5),
                    context.get("expectation_shock", 0.0),
                    context.get("meme_collapse", 0.0),
                    context.get("fictional_belief", 0.5)
                ], dtype=torch.float32).to(self.zombie_net.device).unsqueeze(0)
                
                self.zombie_net.eval()
                with torch.no_grad():
                    output = self.zombie_net(input_data)[0]
                
                zombie_metrics = {
                    "GDP": output[0].item(),
                    "employment": output[1].item(),
                    "trade": output[2].item()
                }
                self.zombie_history.append(zombie_metrics)
                self.reflexive_factor = min(1.0, self.reflexive_factor + 0.1)
                logging.debug(f"ReflexiveNecroeconomics {self.nation}: Zombie state active, metrics {zombie_metrics}")
                return zombie_metrics
            return {}
        except Exception as e:
            logging.error(f"Error in simulate_zombie_economy for {self.nation}: {e}")
            return {}

    def train_zombie_net(self, context: Dict[str, float], actual_metrics: Dict[str, float]):
        try:
            input_data = torch.tensor([
                context.get("pmi", 0.5),
                context.get("market_sentiment", 0.0),
                context.get("fear_index", 0.0),
                context.get("Stock_Volatility", 0.0),
                context.get("inflation", 0.04),
                context.get("trade", 1.0),
                context.get("resilience", 0.5),
                context.get("expectation_shock", 0.0),
                context.get("meme_collapse", 0.0),
                context.get("fictional_belief", 0.5)
            ], dtype=torch.float32).to(self.zombie_net.device).unsqueeze(0)
            
            target = torch.tensor([
                actual_metrics.get("GDP", 0.5),
                actual_metrics.get("employment", 0.5),
                actual_metrics.get("trade", 0.5)
            ], dtype=torch.float32).to(self.zombie_net.device)
            
            self.zombie_net.train()
            self.optimizer.zero_grad()
            output = self.zombie_net(input_data)[0]
            loss = nn.MSELoss()(output, target)
            loss.backward()
            self.optimizer.step()
            logging.debug(f"ReflexiveNecroeconomics {self.nation}: Trained with loss {loss.item():.4f}")
        except Exception as e:
            logging.error(f"Error in train_zombie_net for {self.nation}: {e}")

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "reflexive_factor": self.reflexive_factor,
                "zombie_state": 1.0 if self.zombie_state else 0.0,
                "history_length": len(self.zombie_history)
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Tầng 31: Economic Singularity
class SingularityCore:
    def __init__(self, nation: str, singularity_threshold: int = 100_000_000):
        self.nation = nation
        self.singularity_threshold = singularity_threshold
        self.ecogol = None
        self.singularity_state = False
        self.singularity_history = deque(maxlen=50)
        self.emotion_map = {"panic": 0.0, "greed": 0.0, "hope": 0.0}
        self.motive = "avoid collapse"

    def activate_ecogol(self, agent_count: int, context: Dict[str, float]):
        try:
            if agent_count > self.singularity_threshold and not self.singularity_state:
                self.singularity_state = True
                self.ecogol = {
                    "emotions": {
                        "panic": context.get("fear_index", 0.0),
                        "greed": context.get("market_sentiment", 0.0) * 0.5,
                        "hope": context.get("hope_index", 0.0)
                    },
                    "will_power": random.uniform(0.5, 1.0)
                }
                self.motive = random.choice(["avoid collapse", "maximize growth", "stabilize markets"])
                logging.warning(f"SingularityCore {self.nation}: ECOGOL activated with motive '{self.motive}'")
            elif self.singularity_state:
                self.ecogol["emotions"]["panic"] = context.get("fear_index", 0.0)
                self.ecogol["emotions"]["greed"] = context.get("market_sentiment", 0.0) * 0.5
                self.ecogol["emotions"]["hope"] = context.get("hope_index", 0.0)
                self.ecogol["will_power"] = min(1.0, self.ecogol["will_power"] + 0.05)
                self.singularity_history.append({"emotions": self.ecogol["emotions"].copy(), "motive": self.motive})
        except Exception as e:
            logging.error(f"Error in activate_ecogol for {self.nation}: {e}")

    def apply_ecogol_will(self, space: Dict[str, float]) -> Dict[str, float]:
        try:
            if self.singularity_state:
                will_power = self.ecogol["will_power"]
                if self.motive == "avoid collapse":
                    space["consumption"] *= (1 - will_power * 0.3)
                    space["resilience"] += will_power * 0.2
                elif self.motive == "maximize growth":
                    space["consumption"] *= (1 + will_power * 0.4)
                    space["market_sentiment"] += will_power * 0.3
                elif self.motive == "stabilize markets":
                    space["Stock_Volatility"] = max(0.0, space.get("Stock_Volatility", 0.0) - will_power * 0.2)
                logging.debug(f"SingularityCore {self.nation}: ECOGOL applied will with motive '{self.motive}'")
            return space
        except Exception as e:
            logging.error(f"Error in apply_ecogol_will for {self.nation}: {e}")
            return space

    def get_metrics(self) -> Dict[str, float]:
        try:
            return {
                "singularity_state": 1.0 if self.singularity_state else 0.0,
                "ecogol_panic": self.ecogol["emotions"]["panic"] if self.ecogol else 0.0,
                "ecogol_greed": self.ecogol["emotions"]["greed"] if self.ecogol else 0.0,
                "ecogol_hope": self.ecogol["emotions"]["hope"] if self.ecogol else 0.0,
                "will_power": self.ecogol["will_power"] if self.ecogol else 0.0
            }
        except Exception as e:
            logging.error(f"Error in get_metrics for {self.nation}: {e}")
            return {}

# Cập nhật HyperAgent để hỗ trợ các tầng mới
def enhance_hyper_agent_for_singularity_layers(HyperAgent):
    class EnhancedHyperAgent(HyperAgent):
        def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                     volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                     policy: Optional[Dict[str, float]] = None) -> None:
            try:
                super().interact(agents, global_context, nation_space, volatility_history, gdp_history, 
                                 market_data, policy)
                
                collapse_layer = global_context.get("collapse_layer", EndogenousCollapseSeeds(self.nation))
                necro_layer = global_context.get("necro_layer", ReflexiveNecroeconomics(self.nation))
                singularity_layer = global_context.get("singularity_layer", SingularityCore(self.nation))
                
                # EndogenousCollapseSeeds
                if collapse_layer.collapse_risk > 0.7:
                    self.fear_index = min(1.0, self.fear_index + collapse_layer.collapse_risk * 0.4)
                    self.wealth *= (1 - collapse_layer.collapse_risk * 0.3)
                
                # ReflexiveNecroeconomics
                if necro_layer.zombie_state:
                    zombie_metrics = necro_layer.simulate_zombie_economy(global_context, agents)
                    if zombie_metrics:
                        self.wealth *= (1 + zombie_metrics["GDP"] * 0.1)
                        self.fear_index += 0.2
                
                # SingularityCore
                if singularity_layer.singularity_state:
                    ecogol_emotions = singularity_layer.ecogol["emotions"]
                    if ecogol_emotions["panic"] > 0.7:
                        self.fear_index = min(1.0, self.fear_index + 0.3)
                    if ecogol_emotions["greed"] > 0.7 and hasattr(self, 'portfolio'):
                        self.portfolio["stocks"] = min(1.0, self.portfolio.get("stocks", 0.0) + 0.2)
                        total = sum(self.portfolio.values())
                        if total > 0:
                            self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                
                if hasattr(self, 'inertia'):
                    psych_dict = {
                        "fear_index": self.fear_index,
                        "hope_index": self.hope_index
                    }
                    adjusted_psych = self.inertia.adjust_behavior(psych_dict)
                    self.fear_index = adjusted_psych["fear_index"]
                    self.hope_index = adjusted_psych["hope_index"]
                
                logging.debug(f"HyperAgent {self.id}: Updated with collapse, necro, singularity layers")
            except Exception as e:
                logging.error(f"Error in interact for {self.id}: {e}")

    return EnhancedHyperAgent

# Cập nhật ShadowAgent để hỗ trợ các tầng mới
def enhance_shadow_agent_for_singularity_layers(ShadowAgent):
    class EnhancedShadowAgent(ShadowAgent):
        def move_wealth_to_gold(self, gold_price: float):
            try:
                super().move_wealth_to_gold(gold_price)
                
                collapse_layer = global_context.get("collapse_layer", EndogenousCollapseSeeds(self.nation))
                necro_layer = global_context.get("necro_layer", ReflexiveNecroeconomics(self.nation))
                singularity_layer = global_context.get("singularity_layer", SingularityCore(self.nation))
                
                # EndogenousCollapseSeeds
                if collapse_layer.collapse_risk > 0.7:
                    self.black_market_flow += self.wealth * collapse_layer.collapse_risk * 0.2
                    self.stress_hormone = min(1.0, self.stress_hormone + 0.3) if hasattr(self, 'stress_hormone') else 0.5
                
                # ReflexiveNecroeconomics
                if necro_layer.zombie_state:
                    self.black_market_flow += self.wealth * 0.15
                    self.wealth *= 0.95
                
                # SingularityCore
                if singularity_layer.singularity_state:
                    ecogol_emotions = singularity_layer.ecogol["emotions"]
                    if ecogol_emotions["panic"] > 0.7 and hasattr(self, 'portfolio'):
                        self.portfolio["gold"] = min(1.0, self.portfolio.get("gold", 0.0) + 0.3)
                        total = sum(self.portfolio.values())
                        if total > 0:
                            self.portfolio = {k: v / total for k, v in self.portfolio.items()}
                        self.gold_holdings = self.portfolio.get("gold", 0.0) * self.wealth / gold_price
                        self.cash_holdings = self.portfolio.get("cash", 0.0) * self.wealth
                        self.wealth = self.cash_holdings + self.gold_holdings * gold_price + \
                                     self.portfolio.get("crypto", 0.0) * self.wealth
                        self.activity_log.append({"action": "portfolio_update", "portfolio": self.portfolio})
                
                logging.debug(f"ShadowAgent {self.id}: Updated with collapse, necro, singularity layers")
            except Exception as e:
                logging.error(f"Error in move_wealth_to_gold for {self.id}: {e}")

    return EnhancedShadowAgent

# Tích hợp các tầng mới vào VoTranhAbyssCoreMicro
def integrate_singularity_layers(core, nation_name: str):
    try:
        core.collapse_layer = getattr(core, 'collapse_layer', {})
        core.necro_layer = getattr(core, 'necro_layer', {})
        core.singularity_layer = getattr(core, 'singularity_layer', {})
        
        core.collapse_layer[nation_name] = EndogenousCollapseSeeds(nation_name)
        core.necro_layer[nation_name] = ReflexiveNecroeconomics(nation_name)
        core.singularity_layer[nation_name] = SingularityCore(nation_name)
        
        # Cập nhật HyperAgent
        core.HyperAgent = enhance_hyper_agent_for_singularity_layers(core.HyperAgent)
        for agent in core.agents:
            agent.__class__ = core.HyperAgent
        
        # Cập nhật ShadowAgent nếu có ShadowEconomy
        if hasattr(core, 'shadow_economies') and nation_name in core.shadow_economies:
            core.shadow_economies[nation_name].ShadowAgent = enhance_shadow_agent_for_singularity_layers(
                core.shadow_economies[nation_name].ShadowAgent
            )
            for agent in core.shadow_economies[nation_name].agents:
                agent.__class__ = core.shadow_economies[nation_name].ShadowAgent
        
        logging.info(f"Integrated EndogenousCollapseSeeds, ReflexiveNecroeconomics, SingularityCore for {nation_name}")
    except Exception as e:
        logging.error(f"Error in integrate_singularity_layers for {nation_name}: {e}")

# Cập nhật reflect_economy để bao gồm các tầng mới
def enhanced_reflect_economy_with_singularity_layers(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                                                    R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0):
    try:
        result = VoTranhAbyssCoreMicro.reflect_economy(self, t, observer, space, R_set, nation_name, external_shock)
        
        if nation_name in self.collapse_layer:
            collapse_layer = self.collapse_layer[nation_name]
            necro_layer = self.necro_layer[nation_name]
            singularity_layer = self.singularity_layer[nation_name]
            
            self.global_context["collapse_layer"] = collapse_layer
            self.global_context["necro_layer"] = necro_layer
            self.global_context["singularity_layer"] = singularity_layer
            
            context = {**self.global_context, **space}
            agents = [a for a in self.agents if a.nation == nation_name]
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                agents += self.shadow_economies[nation_name].agents
            
            # EndogenousCollapseSeeds
            collapse_layer.plant_seed(t, context)
            activated_seeds = collapse_layer.check_seeds(t, context)
            if activated_seeds:
                space["consumption"] *= 0.5
                space["market_sentiment"] -= 0.4
                space["fear_index"] += 0.5
                space["resilience"] -= 0.3
                result["Insight"]["Psychology"] += f" | Collapse seeds ({len(activated_seeds)}) detonated, market in turmoil."
            
            # ReflexiveNecroeconomics
            zombie_metrics = necro_layer.simulate_zombie_economy(context, agents)
            if zombie_metrics:
                space["GDP"] = zombie_metrics["GDP"]
                space["trade"] = zombie_metrics["trade"]
                space["resilience"] -= 0.2
                result["Insight"]["Psychology"] += f" | Zombie economy active, GDP {zombie_metrics['GDP']:.3f}."
                necro_layer.train_zombie_net(context, zombie_metrics)
            
            # SingularityCore
            singularity_layer.activate_ecogol(len(agents), context)
            space = singularity_layer.apply_ecogol_will(space)
            if singularity_layer.singularity_state:
                space["consumption"] *= 1.2 if singularity_layer.motive == "maximize growth" else 0.9
                space["market_sentiment"] += 0.3 if singularity_layer.ecogol["emotions"]["hope"] > 0.7 else -0.2
                result["Insight"]["Psychology"] += f" | ECOGOL motive '{singularity_layer.motive}', will power {singularity_layer.ecogol['will_power']:.3f}."
            
            # Ảnh hưởng đến shadow economy
            if hasattr(self, 'shadow_economies') and nation_name in self.shadow_economies:
                shadow_economy = self.shadow_economies[nation_name]
                collapse_metrics = collapse_layer.get_metrics()
                necro_metrics = necro_layer.get_metrics()
                singularity_metrics = singularity_layer.get_metrics()
                shadow_economy.cpi_impact += (collapse_metrics["collapse_risk"] + 
                                             necro_metrics["reflexive_factor"] + 
                                             singularity_metrics["ecogol_panic"]) * 0.15
                shadow_economy.liquidity_pool *= 1.2 if singularity_metrics["singularity_state"] else 1.0
                shadow_economy.tax_loss += collapse_metrics["collapse_risk"] * 0.1 * shadow_economy.liquidity_pool
            
            result["Collapse_Layer"] = collapse_metrics
            result["Necro_Layer"] = necro_metrics
            result["Singularity_Layer"] = singularity_metrics
            self.history[nation_name][-1]["collapse_metrics"] = collapse_metrics
            self.history[nation_name][-1]["necro_metrics"] = necro_metrics
            self.history[nation_name][-1]["singularity_metrics"] = singularity_metrics
        
        return result
    except Exception as e:
        logging.error(f"Error in enhanced_reflect_economy_with_singularity_layers for {nation_name}: {e}")
        return result

# Gắn hàm enhanced_reflect_economy_with_singularity_layers vào class VoTranhAbyssCoreMicro
setattr(VoTranhAbyssCoreMicro, 'reflect_economy', enhanced_reflect_economy_with_singularity_layers)

# Xuất dữ liệu các tầng
def export_singularity_layers_data(core, nation_name: str):
    try:
        if hasattr(core, 'collapse_layer') and nation_name in core.collapse_layer:
            collapse = core.collapse_layer[nation_name]
            data = {
                "Seed_ID": [s.id for s in collapse.seeds],
                "Planted_Step": [s.planted_step for s in collapse.seeds],
                "Activated": [1 if s.activated else 0 for s in collapse.seeds]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"collapse_seeds_{nation_name}.csv", index=False)
        
        if hasattr(core, 'necro_layer') and nation_name in core.necro_layer:
            necro = core.necro_layer[nation_name]
            data = {
                "GDP": [h["GDP"] for h in necro.zombie_history],
                "Employment": [h["employment"] for h in necro.zombie_history],
                "Trade": [h["trade"] for h in necro.zombie_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"reflexive_necro_{nation_name}.csv", index=False)
        
        if hasattr(core, 'singularity_layer') and nation_name in core.singularity_layer:
            singularity = core.singularity_layer[nation_name]
            data = {
                "Step": list(range(len(singularity.singularity_history))),
                "Panic": [h["emotions"]["panic"] for h in singularity.singularity_history],
                "Greed": [h["emotions"]["greed"] for h in singularity.singularity_history],
                "Hope": [h["emotions"]["hope"] for h in singularity.singularity_history],
                "Motive": [h["motive"] for h in singularity.singularity_history]
            }
            df = pd.DataFrame(data)
            df.to_csv(f"economic_singularity_{nation_name}.csv", index=False)
        
        logging.info(f"Exported data for layers 29-31 in {nation_name}")
    except Exception as e:
        logging.error(f"Error in export_singularity_layers_data for {nation_name}: {e}")

# Ví dụ sử dụng
if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}}
    ]
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12")
    
    integrate_shadow_economy(core, "Vietnam")
    integrate_cultural_inertia(core, "Vietnam")
    integrate_propaganda_layer(core, "Vietnam")
    integrate_multiverse_simulator(core, "Vietnam")
    integrate_trust_dynamics(core, "Vietnam")
    integrate_timewarp_gdp(core, "Vietnam")
    integrate_neocortex_emulator(core, "Vietnam")
    integrate_shaman_council(core, "Vietnam")
    integrate_self_awareness(core, "Vietnam")
    integrate_investment_inertia(core, "Vietnam")
    integrate_mnemonic_market(core, "Vietnam")
    integrate_expectation_decay(core, "Vietnam")
    integrate_nostalgia_portfolio(core, "Vietnam")
    integrate_illusion_grid(core, "Vietnam")
    integrate_echo_chamber(core, "Vietnam")
    integrate_possession_layer(core, "Vietnam")
    integrate_parallel_leak(core, "Vietnam")
    integrate_singularity_layers(core, "Vietnam")
    
    result = core.reflect_economy(
        t=1.0,
        observer=core.nations["Vietnam"]["observer"],
        space=core.nations["Vietnam"]["space"],
        R_set=[{"growth": 0.03, "cash_flow": 0.5}],
        nation_name="Vietnam"
    )
    
    export_singularity_layers_data(core, "Vietnam")
    print(f"Collapse Layer Metrics: {result.get('Collapse_Layer', {})}")
    print(f"Necro Layer Metrics: {result.get('Necro_Layer', {})}")
    print(f"Singularity Layer Metrics: {result.get('Singularity_Layer', {})}")
