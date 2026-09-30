import numpy as np
import pandas as pd

def generate_hourly_events(demand, service_mean_minutes=14.73, service_std_minutes=6.47, seed=None):

    if demand <= 0:
        return np.array([]), np.array([])
        
    if seed is not None:
        np.random.seed(seed)
        
    # Berdasarkan Teorema Proses Poisson: jika jumlah kedatangan (demand) dalam 
    # interval [0, T] diketahui, waktu kedatangannya terdistribusi Uniform(0, T).
    arrival_times = np.sort(np.random.uniform(0, 60, int(demand)))
    
    # Menghitung parameter distribusi Lognormal dari Mean dan Std Dev aktual
    var = service_std_minutes**2
    mean_sq = service_mean_minutes**2
    
    # Formula konversi ruang linear ke ruang logaritmik
    sigma_log = np.sqrt(np.log(1 + (var / mean_sq)))
    mu_log = np.log(service_mean_minutes) - (sigma_log**2 / 2)
    
    # Generate service time (waktu persiapan restoran)
    service_times = np.random.lognormal(mean=mu_log, sigma=sigma_log, size=int(demand))
    
    return arrival_times, service_times

def simulate_restaurant_queue(arrival_times, service_times, num_staff):

    if len(arrival_times) == 0:
        return {'avg_wait_mins': 0, 'p95_wait_mins': 0, 'max_wait_mins': 0, 'utilization': 0}
        
    staff_available_time = np.zeros(num_staff)
    waiting_times = np.zeros(len(arrival_times))
    queue_lengths = [] # 
    
    for i in range(len(arrival_times)):
        arrival = arrival_times[i]
        service = service_times[i]
        
        current_queue = sum(1 for t in staff_available_time if t > arrival)
        queue_lengths.append(max(0, current_queue - num_staff))
        
        earliest_staff_idx = np.argmin(staff_available_time)
        earliest_available = staff_available_time[earliest_staff_idx]
        
        if earliest_available > arrival:
            wait = earliest_available - arrival
            start_time = earliest_available
        else:
            wait = 0
            start_time = arrival
            
        waiting_times[i] = wait
        staff_available_time[earliest_staff_idx] = start_time + service
        
    avg_wait = np.mean(waiting_times)
    p95_wait = np.percentile(waiting_times, 95)
    max_wait = np.max(waiting_times)
    
    total_service_time = np.sum(service_times)
    total_capacity = num_staff * 60
    utilization = min(total_service_time / total_capacity, 1.0)
    
    return {
        'avg_wait_mins': round(avg_wait, 2),
        'p95_wait_mins': round(p95_wait, 2),
        'max_wait_mins': round(max_wait, 2),
        'utilization': round(utilization, 2)
    }

if __name__ == "__main__":
    test_demand = 7
    
    print(f"Simulation demand: {test_demand} orders/hour")
    print("Menggunakan base stat: Mean 14.73 menit, Std Dev 6.47 menit\n")
    
    arr, srv = generate_hourly_events(test_demand, seed=10)
    
    print("-" * 65)
    print(f"{'Staff (c)':<10} | {'Avg Wait':<12} | {'P95 Wait':<12} | {'Utilisasi'}")
    print("-" * 65)
    
    for c in range(5, 15):
        metrics = simulate_restaurant_queue(arr, srv, num_staff=c)
        print(f"{c:<10} | {metrics['avg_wait_mins']:<12} | {metrics['p95_wait_mins']:<12} | {metrics['utilization']}")