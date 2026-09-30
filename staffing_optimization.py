import pandas as pd
import numpy as np
import joblib
import importlib
from tqdm import tqdm

queue_sim = importlib.import_module("queue_simulation")
model = joblib.load('model_demand_pipeline.pkl')

# Parameter SLA and Baseline
SLA_MAX_WAIT_MINS = 10.0
MAX_STAFF_LIMIT = 20
FIXED_STAFF_COUNT = 8

# 2. LOAD DATA TEST (Use 1 week of data from the busiest restaurants)
print("Loading and preparing evaluation data...")
df = pd.read_csv('cleaned_hourly_demand_final.csv')
df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
df = df.sort_values(by=['restaurant_id', 'order_timestamp']).dropna().reset_index(drop=True)

# Get data one week for the top 3 busiest restaurants
top_restaurants = df['restaurant_id'].value_counts().head(3).index
eval_df = df[df['restaurant_id'].isin(top_restaurants)].copy()
eval_df = eval_df.tail(24 * 7 * len(top_restaurants))

# Predicted Demand
eval_df['predicted_demand'] = np.ceil(model.predict(eval_df)).astype(int)

eval_df = eval_df[eval_df['demand'] > 0].copy()

def evaluate_staffing_policy(row, policy_type):
    actual_demand = int(row['demand'])
    pred_demand = int(row['predicted_demand'])
    
    arr, srv = queue_sim.generate_hourly_events(demand=actual_demand)
    
    if policy_type == 'fixed':
        metrics = queue_sim.simulate_restaurant_queue(arr, srv, num_staff=FIXED_STAFF_COUNT)
        return FIXED_STAFF_COUNT, metrics['p95_wait_mins']
        
    elif policy_type == 'dynamic_ml':
        arr_pred, srv_pred = queue_sim.generate_hourly_events(demand=pred_demand)
        
        optimal_staff = None
        for c in range(1, MAX_STAFF_LIMIT + 1):
            pred_metrics = queue_sim.simulate_restaurant_queue(arr_pred, srv_pred, num_staff=c)
            if pred_metrics['p95_wait_mins'] <= SLA_MAX_WAIT_MINS:
                optimal_staff = c
                break
                
        if optimal_staff is None:
            optimal_staff = MAX_STAFF_LIMIT
            
        actual_metrics = queue_sim.simulate_restaurant_queue(arr, srv, num_staff=optimal_staff)
        return optimal_staff, actual_metrics['p95_wait_mins']

print("Running comparative simulation for staffing policies...")
results = []

for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0]):
    rest_id = row['restaurant_id']
    timestamp = row['order_timestamp']
    actual_demand = row['demand']
    
    # Uji Kebijakan 1: Fixed Staffing
    fixed_staff, fixed_p95 = evaluate_staffing_policy(row, 'fixed')
    
    # Uji Kebijakan 2: Dynamic Staffing (ML)
    dyn_staff, dyn_p95 = evaluate_staffing_policy(row, 'dynamic_ml')
    
    results.append({
        'restaurant_id': rest_id,
        'timestamp': timestamp,
        'actual_demand': actual_demand,
        'fixed_staff': fixed_staff,
        'fixed_p95_wait': fixed_p95,
        'fixed_sla_violation': 1 if fixed_p95 > SLA_MAX_WAIT_MINS else 0,
        'dynamic_staff': dyn_staff,
        'dynamic_p95_wait': dyn_p95,
        'dynamic_sla_violation': 1 if dyn_p95 > SLA_MAX_WAIT_MINS else 0
    })

results_df = pd.DataFrame(results)

# 5. REKAPITULASI METRIK EVALUASI
print("\n=== Results staffing optimization ===")
summary = pd.DataFrame({
    'Metrik': ['Total Staff-Hours', 'Mean Staff per Hour', 'Total SLA Violation (hours)', 'mean P95 Wait (mins)'],
    'Fixed Staffing': [
        results_df['fixed_staff'].sum(),
        results_df['fixed_staff'].mean(),
        results_df['fixed_sla_violation'].sum(),
        results_df['fixed_p95_wait'].mean()
    ],
    'Dynamic Staffing (ML)': [
        results_df['dynamic_staff'].sum(),
        results_df['dynamic_staff'].mean(),
        results_df['dynamic_sla_violation'].sum(),
        results_df['dynamic_p95_wait'].mean()
    ]
})

print(summary.to_string(index=False))

results_df.to_csv('optimization_results.csv', index=False)
print("\n Result saved as 'optimization_results.csv'")