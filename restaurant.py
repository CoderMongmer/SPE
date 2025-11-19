import simpy
import random
import numpy as np
import pandas as pd
import itertools

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
class SystemConfig:
    STATIONS = ['Drink', 'Salad', 'Main', 'Dessert']
    SERVICE_TIMES_SEC = {'Drink': 20, 'Salad': 45, 'Main': 60, 'Dessert': 30}
    SERVICE_RATES_MU = {k: 60.0 / v for k, v in SERVICE_TIMES_SEC.items()}
    P_INITIAL = np.array([0.20, 0.30, 0.40, 0.10])
    P_MATRIX = np.array([
        [0.00, 0.40, 0.50, 0.05],
        [0.10, 0.00, 0.70, 0.05],
        [0.15, 0.05, 0.00, 0.50],
        [0.05, 0.00, 0.05, 0.00]
    ])

    @staticmethod
    def calculate_theoretical_rho(alpha, capacities):
        a = alpha * SystemConfig.P_INITIAL
        I = np.eye(4)
        try:
            lambda_eff = np.linalg.inv(I - SystemConfig.P_MATRIX.T).dot(a)
        except:
            lambda_eff = np.zeros(4)

        results = {}
        for i, name in enumerate(SystemConfig.STATIONS):
            c = capacities[name]
            mu = SystemConfig.SERVICE_RATES_MU[name]
            results[name] = lambda_eff[i] / (c * mu) if c > 0 else 999.9
        return results

    @staticmethod
    def get_status(rho):
        if rho >= 0.99: return "Unstable"
        if rho >= 0.90: return "Critical"
        return "Stable"


# ==========================================
# 2. SIMULATION MODEL
# ==========================================
class FoodStation:
    def __init__(self, env, name, capacity, service_time_sec):
        self.env = env
        self.name = name
        self.resource = simpy.Resource(env, capacity=capacity)
        self.mean_service_time_min = service_time_sec / 60.0
        self.total_busy_time = 0.0
        self.wait_times = []
        self.queue_area = 0.0
        self.last_update_time = 0.0

    def update_lq(self):
        curr = self.env.now
        self.queue_area += len(self.resource.queue) * (curr - self.last_update_time)
        self.last_update_time = curr

    def serve(self, customer_id):
        arr = self.env.now
        self.update_lq()
        with self.resource.request() as req:
            yield req
            self.update_lq()

            wait_duration = self.env.now - arr
            self.wait_times.append(wait_duration)

            service_duration = random.expovariate(1.0 / self.mean_service_time_min)
            yield self.env.timeout(service_duration)

            self.total_busy_time += service_duration


class BuffetSimulation:
    def __init__(self, alpha, capacities):
        self.env = simpy.Environment()
        self.alpha = alpha
        self.stations = {n: FoodStation(self.env, n, capacities[n], SystemConfig.SERVICE_TIMES_SEC[n]) for n in
                         SystemConfig.STATIONS}
        self.customer_times = []

    def customer_process(self, cust_id):
        start = self.env.now
        curr_idx = np.random.choice(range(4), p=SystemConfig.P_INITIAL)
        while True:
            yield self.env.process(self.stations[SystemConfig.STATIONS[curr_idx]].serve(cust_id))
            probs = list(SystemConfig.P_MATRIX[curr_idx])
            probs.append(1.0 - sum(probs))
            next_step = np.random.choice(range(5), p=probs)
            if next_step == 4: break
            curr_idx = next_step
        self.customer_times.append(self.env.now - start)

    def run(self, runtime=480, warmup=120):
        def gen():
            i = 0
            while True:
                yield self.env.timeout(random.expovariate(self.alpha))
                i += 1
                self.env.process(self.customer_process(i))

        self.env.process(gen())

        self.env.run(until=warmup)

        self.customer_times = []
        for s in self.stations.values():
            s.total_busy_time = 0
            s.wait_times = []
            s.queue_area = 0
            s.last_update_time = self.env.now

        self.env.run(until=runtime)

        for s in self.stations.values(): s.update_lq()

        actual_time = runtime - warmup
        res = {}
        for n, s in self.stations.items():
            res[n] = {
                'rho': s.total_busy_time / (s.resource.capacity * actual_time),
                'wq': np.mean(s.wait_times) if s.wait_times else 0,
                'lq': s.queue_area / actual_time
            }
        return res, np.mean(self.customer_times) if self.customer_times else 0


# ==========================================
# 3. EXECUTION
# ==========================================

def print_header():
    # Primary Header
    print("-" * 230)
    header_str = f"{'Ph':<3} | {'ID':<3} | {'Alp':<3} | {'W_Sys':<6} || "
    for st in SystemConfig.STATIONS:
        header_str += f"{st.upper():<42} || "
    print(header_str)

    # Secondary Header (Metrics)
    sub_header = f"{'':<3} | {'':<3} | {'':<3} | {'(min)':<6} || "
    sub_cols = f"{'Svr':<3} {'RhoT':<5} {'RhoS':<5} {'Wq':<5} {'Lq':<5} {'Status':<8} | "
    for _ in SystemConfig.STATIONS:
        sub_header += sub_cols + "| "
    print(sub_header)
    print("-" * 230)


def run_scenario(sc_id, alpha, caps, phase_name):
    theo = SystemConfig.calculate_theoretical_rho(alpha, caps)
    sim = BuffetSimulation(alpha, caps)
    res, w_sys = sim.run(runtime=480, warmup=120)

    row_str = f"{phase_name:<3} | {sc_id:<3} | {alpha:<3} | {w_sys:<6.2f} || "
    is_stable = True

    for st in SystemConfig.STATIONS:
        svr = caps[st]
        rt = theo[st]
        rs = res[st]['rho']
        wq = res[st]['wq']
        lq = res[st]['lq']
        stat = SystemConfig.get_status(rs)

        if rs >= 1.0: is_stable = False

        short_st = "Unstbl" if stat == "Unstable" else "Crit" if stat == "Critical" else "Stable"

        # Format: Svr RhoT RhoS Wq Lq Status
        row_str += f"{svr:<3} {rt:<5.2f} {rs:<5.2f} {wq:<5.2f} {lq:<5.2f} {short_st:<8} | | "

    print(row_str)

    return {
        'ID': sc_id, 'Alpha': alpha, 'Caps': caps,
        'W_Sys': w_sys, 'Is_Stable': is_stable, 'Results': res
    }


def run_two_phase():
    print("\n=== Phase 1: Sorting (32 Scenarios - Min/Max Levels) ===")
    print_header()

    p1_factors = {
        'alpha': [2, 10], 'c_main': [8, 14],
        'c_salad': [4, 6], 'c_dessert': [2, 4], 'c_drink': [1, 4]
    }
    keys, values = zip(*p1_factors.items())
    p1_scenarios = [dict(zip(keys, v)) for v in itertools.product(*values)]

    p1_results = []
    for idx, sc in enumerate(p1_scenarios):
        caps = {'Drink': sc['c_drink'], 'Salad': sc['c_salad'], 'Main': sc['c_main'], 'Dessert': sc['c_dessert']}
        res = run_scenario(idx + 1, sc['alpha'], caps, "P1")
        p1_results.append(res)

    # Analyze P1 results for phase 2
    df_p1 = pd.DataFrame(p1_results)
    candidates = df_p1[(df_p1['Alpha'] == 10) & (df_p1['Is_Stable'] == True)]

    if candidates.empty:
        print("\n[INFO] No Stable configuration found at Alpha=10 in P1. Selecting best W_Sys candidate.")
        best_p1 = df_p1[df_p1['Alpha'] == 10].sort_values('W_Sys').iloc[0]
    else:
        best_p1 = candidates.sort_values('W_Sys').iloc[0]

    print(f"\n=== PHASE 2: REFINEMENT (8 Scenarios - Focusing on Main/Salad) ===")
    print_header()

    fixed_alpha = 10
    fixed_drink = best_p1['Caps']['Drink']
    fixed_dessert = best_p1['Caps']['Dessert']

    # Focus on all levels for critical factors (Main, Salad)
    p2_factors = {'c_main': [8, 10, 12, 14], 'c_salad': [4, 6]}
    keys2, values2 = zip(*p2_factors.items())
    p2_scenarios = [dict(zip(keys2, v)) for v in itertools.product(*values2)]

    p2_results = []
    for idx, sc in enumerate(p2_scenarios):
        caps = {
            'Drink': fixed_drink, 'Salad': sc['c_salad'],
            'Main': sc['c_main'], 'Dessert': fixed_dessert
        }
        res = run_scenario(32 + idx + 1, fixed_alpha, caps, "P2")
        p2_results.append(res)

    df_p2 = pd.DataFrame(p2_results)
    stable_p2 = df_p2[df_p2['Is_Stable'] == True]

    print("\n" + "=" * 80)
    if not stable_p2.empty:
        best = stable_p2.sort_values('W_Sys').iloc[0]
        print(f"OPTIMAL SCENARIO: ID {best['ID']}")
        print(
            f"Configuration: Alpha={best['Alpha']}, Drink={best['Caps']['Drink']}, Salad={best['Caps']['Salad']}, Main={best['Caps']['Main']}, Dessert={best['Caps']['Dessert']}")
        print(f"Performance: W_Sys = {best['W_Sys']:.2f} minutes")
    else:
        print("There is no best scenario")
    print("=" * 80)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_two_phase()