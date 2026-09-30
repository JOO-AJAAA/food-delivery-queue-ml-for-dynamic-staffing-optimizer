import pandas as pd
import numpy as np
import joblib
import importlib
from tqdm import tqdm

queue_sim = importlib.import_module("queue_simulation")
model = joblib.load('model_demand_pipeline.pkl')

MAX_STAFF_LIMIT = 20
FIXED_STAFF_COUNT = 8
SLA_SCENARIOS = [5.0, 10.0, 15.0] 

df = pd.read_csv('cleaned_hourly_demand_final.csv')
df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
eval_df = df.sort_values(by=['restaurant_id', 'order_timestamp']).dropna().reset_index(drop=True)

top_restaurants = eval_df['restaurant_id'].value_counts().head(3).index
eval_df = eval_df[eval_df['restaurant_id'].isin(top_restaurants)].tail(24 * 7 * len(top_restaurants)).copy()
eval_df['predicted_demand'] = np.ceil(model.predict(eval_df)).astype(int)
eval_df = eval_df[eval_df['demand'] > 0].copy()

experiment_results = []

print("Run Sensitivity Analysis for various SLA...")
for sla_target in SLA_SCENARIOS:
    print(f"\ Evaluating SLA: {sla_target} mins")
    
    total_dynamic_staff = 0
    total_dynamic_violations = 0
    total_fixed_staff = 0
    total_fixed_violations = 0
    
    for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0]):
        actual_demand = int(row['demand'])
        pred_demand = int(row['predicted_demand'])
        
        arr, srv = queue_sim.generate_hourly_events(demand=actual_demand)
        arr_pred, srv_pred = queue_sim.generate_hourly_events(demand=pred_demand)
        
        # 1. Fixed Staffing (Baseline)
        fixed_metrics = queue_sim.simulate_restaurant_queue(arr, srv, num_staff=FIXED_STAFF_COUNT)
        total_fixed_staff += FIXED_STAFF_COUNT
        if fixed_metrics['p95_wait_mins'] > sla_target:
            total_fixed_violations += 1
            
        # 2. Dynamic Staffing
        optimal_staff = None
        for c in range(1, MAX_STAFF_LIMIT + 1):
            pred_metrics = queue_sim.simulate_restaurant_queue(arr_pred, srv_pred, num_staff=c)
            if pred_metrics['p95_wait_mins'] <= sla_target:
                optimal_staff = c
                break
                
        if optimal_staff is None:
            optimal_staff = MAX_STAFF_LIMIT
            
        actual_metrics = queue_sim.simulate_restaurant_queue(arr, srv, num_staff=optimal_staff)
        total_dynamic_staff += optimal_staff
        if actual_metrics['p95_wait_mins'] > sla_target:
            total_dynamic_violations += 1

    experiment_results.append({
        'SLA_Target_Mins': sla_target,
        'Fixed_Total_Staff_Hours': total_fixed_staff,
        'Fixed_Violations': total_fixed_violations,
        'Dynamic_Total_Staff_Hours': total_dynamic_staff,
        'Dynamic_Violations': total_dynamic_violations,
        'Staff_Hours_Saved': total_fixed_staff - total_dynamic_staff
    })

exp_df = pd.DataFrame(experiment_results)
exp_df.to_csv('experiment_sla_results.csv', index=False)
print("\n Results saved as 'experiment_sla_results.csv'")