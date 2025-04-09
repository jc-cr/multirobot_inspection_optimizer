import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from multiprocessing import Pool
import itertools

from solver import solve_multi_robot_inspection_problem

def generate_random_waypoints(n, min_coord=0, max_coord=100):
    """Generate n random waypoints within the specified coordinate range."""
    return [(random.uniform(min_coord, max_coord), random.uniform(min_coord, max_coord)) for _ in range(n)]

def run_performance_test(params):
    """Run a single test with given parameters and record performance metrics."""

    (n_waypoints, n_aerial, n_ground, aerial_speed, ground_speed, 
     aerial_max_time, ground_max_time, aerial_inspection_time, 
     ground_inspection_time) = params
    
    # Generate random waypoints
    waypoints = generate_random_waypoints(n_waypoints)
    
    # Set depot locations
    aerial_depot = (0, 0)
    ground_depot = (100, 0)
    
    # Measure solution time
    start_time = time.time()
    
    # Solve the problem
    solution = solve_multi_robot_inspection_problem(
        waypoints, aerial_depot, ground_depot,
        aerial_speed, ground_speed,
        aerial_max_time, ground_max_time,
        aerial_inspection_time, ground_inspection_time,
        n_aerial, n_ground
    )
    
    solve_time = time.time() - start_time
    
    # Extract relevant metrics
    if solution["status"] == "optimal":
        objective_value = solution["objective_value"]
        
        # Calculate utilization metrics
        aerial_used = sum(1 for k in range(n_aerial) if any(solution["aerial_visited"][k].values()))
        ground_used = sum(1 for l in range(n_ground) if any(solution["ground_visited"][l].values()))
        
        # Calculate average route lengths
        aerial_distances = [solution["aerial_distances"].get(k, 0) for k in range(n_aerial) if any(solution["aerial_visited"][k].values())]
        ground_distances = [solution["ground_distances"].get(l, 0) for l in range(n_ground) if any(solution["ground_visited"][l].values())]
        
        aerial_avg_distance = np.mean(aerial_distances) if aerial_distances else 0
        ground_avg_distance = np.mean(ground_distances) if ground_distances else 0
        
        # Calculate average route times
        aerial_times = [solution["total_aerial_times"].get(k, 0) for k in range(n_aerial) if any(solution["aerial_visited"][k].values())]
        ground_times = [solution["total_ground_times"].get(l, 0) for l in range(n_ground) if any(solution["ground_visited"][l].values())]
        
        aerial_avg_time = np.mean(aerial_times) if aerial_times else 0
        ground_avg_time = np.mean(ground_times) if ground_times else 0
        
    else:
        objective_value = 0
        aerial_used = 0
        ground_used = 0
        aerial_avg_distance = 0
        ground_avg_distance = 0
        aerial_avg_time = 0
        ground_avg_time = 0
    
    return {
        "n_waypoints": n_waypoints,
        "n_aerial": n_aerial,
        "n_ground": n_ground,
        "aerial_speed": aerial_speed,
        "ground_speed": ground_speed,
        "aerial_max_time": aerial_max_time,
        "ground_max_time": ground_max_time,
        "aerial_inspection_time": aerial_inspection_time,
        "ground_inspection_time": ground_inspection_time,
        "solve_time": solve_time,
        "status": solution["status"],
        "objective_value": objective_value,
        "aerial_used": aerial_used,
        "ground_used": ground_used,
        "aerial_avg_distance": aerial_avg_distance,
        "ground_avg_distance": ground_avg_distance,
        "aerial_avg_time": aerial_avg_time,
        "ground_avg_time": ground_avg_time,
    }

def run_performance_tests(results_save_path):
    """Run performance tests with varying parameters."""
    os.makedirs(results_save_path, exist_ok=True)
    
    # Define parameter ranges for testing
    n_waypoints_range = [5, 10, 15, 20, 25]
    n_aerial_range = [1, 2, 3]
    n_ground_range = [1, 2, 3]
    
    # Fixed parameters for this test
    aerial_speed = 5.0
    ground_speed = 2.0
    aerial_max_time = 30.0
    ground_max_time = 60.0
    aerial_inspection_time = 1.0
    ground_inspection_time = 3.0
    
    # Generate all parameter combinations
    param_combinations = []
    for n_waypoints in n_waypoints_range:
        for n_aerial in n_aerial_range:
            for n_ground in n_ground_range:
                params = (n_waypoints, n_aerial, n_ground, aerial_speed, ground_speed,
                         aerial_max_time, ground_max_time, aerial_inspection_time,
                         ground_inspection_time)
                param_combinations.append(params)
    
    # Run tests in parallel
    with Pool() as pool:
        results = pool.map(run_performance_test, param_combinations)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(f"{results_save_path}/performance_results.csv", index=False)
    
    return results_df

def run_sensitivity_analysis(results_save_path):
    """Run sensitivity analysis on key parameters."""

    os.makedirs(results_save_path, exist_ok=True)
    
    # Base case parameters
    base_n_waypoints = 15
    base_n_aerial = 2
    base_n_ground = 2
    base_aerial_speed = 5.0
    base_ground_speed = 2.0
    base_aerial_max_time = 30.0
    base_ground_max_time = 60.0
    base_aerial_inspection_time = 1.0
    base_ground_inspection_time = 3.0
    
    # Sensitivity tests
    sensitivity_tests = [
        # Vary aerial speed
        [(base_n_waypoints, base_n_aerial, base_n_ground, speed, base_ground_speed,
          base_aerial_max_time, base_ground_max_time, base_aerial_inspection_time,
          base_ground_inspection_time) for speed in [3.0, 4.0, 5.0, 6.0, 7.0]],
        
        # Vary ground speed
        [(base_n_waypoints, base_n_aerial, base_n_ground, base_aerial_speed, speed,
          base_aerial_max_time, base_ground_max_time, base_aerial_inspection_time,
          base_ground_inspection_time) for speed in [1.0, 1.5, 2.0, 2.5, 3.0]],
        
        # Vary aerial max time
        [(base_n_waypoints, base_n_aerial, base_n_ground, base_aerial_speed, base_ground_speed,
          max_time, base_ground_max_time, base_aerial_inspection_time,
          base_ground_inspection_time) for max_time in [20.0, 25.0, 30.0, 35.0, 40.0]],
        
        # Vary ground max time
        [(base_n_waypoints, base_n_aerial, base_n_ground, base_aerial_speed, base_ground_speed,
          base_aerial_max_time, max_time, base_aerial_inspection_time,
          base_ground_inspection_time) for max_time in [40.0, 50.0, 60.0, 70.0, 80.0]],
        
        # Vary aerial inspection time
        [(base_n_waypoints, base_n_aerial, base_n_ground, base_aerial_speed, base_ground_speed,
          base_aerial_max_time, base_ground_max_time, insp_time,
          base_ground_inspection_time) for insp_time in [0.5, 1.0, 1.5, 2.0, 2.5]],
        
        # Vary ground inspection time
        [(base_n_waypoints, base_n_aerial, base_n_ground, base_aerial_speed, base_ground_speed,
          base_aerial_max_time, base_ground_max_time, base_aerial_inspection_time,
          insp_time) for insp_time in [1.0, 2.0, 3.0, 4.0, 5.0]],
    ]
    
    # Flatten the list of parameter combinations
    param_combinations = list(itertools.chain(*sensitivity_tests))
    
    # Run tests for all parameter combinations
    all_results = []
    
    # Run 3 tests with different random waypoints for each parameter combination
    for params in param_combinations:
        for _ in range(3):
            result = run_performance_test(params)
            all_results.append(result)
    
    # Convert to DataFrame
    sensitivity_df = pd.DataFrame(all_results)
    
    # Save to CSV
    sensitivity_df.to_csv(f"{results_save_path}/sensitivity_results.csv", index=False)
    
    return sensitivity_df

def analyze_and_plot_results(performance_df, sensitivity_df, results_save_path):
    """Analyze results and create plots for the report."""
    
    # make path
    os.makedirs(results_save_path, exist_ok=True)
    
    
    # Plot 1: Problem size vs. solve time
    plt.figure(figsize=(10, 6))
    for n_aerial in performance_df['n_aerial'].unique():
        for n_ground in performance_df['n_ground'].unique():
            subset = performance_df[(performance_df['n_aerial'] == n_aerial) & 
                                   (performance_df['n_ground'] == n_ground)]
            if not subset.empty:
                plt.plot(subset['n_waypoints'], subset['solve_time'], 
                         marker='o', label=f'Aerial: {n_aerial}, Ground: {n_ground}')
    
    plt.xlabel('Number of Waypoints')
    plt.ylabel('Solve Time (seconds)')
    plt.title('Problem Size vs. Solve Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_save_path}/problem_size_vs_solve_time.png', dpi=300)
    
    # Plot 2: Problem size vs. objective value
    plt.figure(figsize=(10, 6))
    for n_aerial in performance_df['n_aerial'].unique():
        for n_ground in performance_df['n_ground'].unique():
            subset = performance_df[(performance_df['n_aerial'] == n_aerial) & 
                                   (performance_df['n_ground'] == n_ground)]
            if not subset.empty:
                plt.plot(subset['n_waypoints'], subset['objective_value'], 
                         marker='o', label=f'Aerial: {n_aerial}, Ground: {n_ground}')
    
    plt.xlabel('Number of Waypoints')
    plt.ylabel('Waypoints Visited')
    plt.title('Problem Size vs. Waypoints Visited')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_save_path}/problem_size_vs_objective.png', dpi=300)
    
    # Sensitivity Analysis Plots
    param_names = ['aerial_speed', 'ground_speed', 'aerial_max_time', 
                   'ground_max_time', 'aerial_inspection_time', 'ground_inspection_time']
    
    for param in param_names:
        plt.figure(figsize=(10, 6))
        # Group by parameter and calculate average objective value
        grouped = sensitivity_df.groupby(param)['objective_value'].mean().reset_index()
        plt.plot(grouped[param], grouped['objective_value'], marker='o')
        
        plt.xlabel(param.replace('_', ' ').title())
        plt.ylabel('Average Waypoints Visited')
        plt.title(f'Effect of {param.replace("_", " ").title()} on Solution Quality')
        plt.grid(True)
        plt.savefig(f'{results_save_path}/sensitivity_{param}.png', dpi=300)
    
    
    return

def main():
    """Main function to run all tests and generate plots."""
    save_path = "analysis_results"
    
    print("Running performance tests...")
    performance_df = run_performance_tests(save_path)
    
    print("Running sensitivity analysis...")
    sensitivity_df = run_sensitivity_analysis(save_path)
    
    print("Generating plots...")
    analyze_and_plot_results(performance_df, sensitivity_df, save_path)
    
    print("Done! Results saved to CSV files and plots saved as PNG images.")

if __name__ == "__main__":
    main()