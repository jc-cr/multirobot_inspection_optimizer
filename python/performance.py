# performance_analysis.py
"""
Computational performance analysis for multi-robot inspection MILP model
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from solver import solve_multi_robot_inspection_problem

def generate_waypoints(map_size, num_points):
    """
    Generate random waypoints within a square map of the specified size.
    
    Args:
        map_size: Size of the map (square, map_size x map_size)
        num_points: Number of waypoints to generate
        
    Returns:
        waypoints: List of waypoint coordinates
        aerial_depot: Coordinates of aerial depot
        ground_depot: Coordinates of ground depot
    """
    # Place depots at opposite corners
    aerial_depot = (0, 0)  # Bottom left
    ground_depot = (map_size, map_size)  # Top right
    
    # Generate random waypoints
    waypoints = []
    for _ in range(num_points):
        x = np.random.uniform(0, map_size)
        y = np.random.uniform(0, map_size)
        waypoints.append((x, y))
    
    return waypoints, aerial_depot, ground_depot

def measure_performance(map_size, num_points, base_params):
    """
    Measure performance of the solver for a specific map size and number of points.
    
    Args:
        map_size: Size of the map
        num_points: Number of waypoints
        base_params: Base parameters for the solver
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    print(f"\nTesting with map_size={map_size}, num_points={num_points}")
    
    # Generate waypoints
    waypoints, aerial_depot, ground_depot = generate_waypoints(map_size, num_points)
    
    # Track time
    start_time = time.time()
    
    # Solve the problem
    try:
        solution = solve_multi_robot_inspection_problem(
            waypoints, aerial_depot, ground_depot,
            base_params['aerial_speed'], 
            base_params['ground_speed'],
            base_params['aerial_max_time'], 
            base_params['ground_max_time'],
            base_params['aerial_inspection_time'], 
            base_params['ground_inspection_time'],
            base_params['num_aerial_robots'], 
            base_params['num_ground_robots']
        )
        
        solve_time = time.time() - start_time
        status = solution["status"]
        
        # Calculate metrics if optimal
        if status == "optimal":
            objective_value = solution['objective_value']
            aerial_waypoints = sum(sum(robot.values()) for robot in solution['aerial_visited'].values())
            ground_waypoints = sum(sum(robot.values()) for robot in solution['ground_visited'].values())
            
            # Calculate other metrics
            aerial_robots_used = sum(1 for k, visited in solution['aerial_visited'].items() 
                                     if sum(visited.values()) > 0)
            ground_robots_used = sum(1 for l, visited in solution['ground_visited'].items() 
                                     if sum(visited.values()) > 0)
            
            metrics = {
                'map_size': map_size,
                'num_points': num_points,
                'solve_time': solve_time,
                'status': status,
                'objective_value': objective_value,
                'aerial_waypoints': aerial_waypoints,
                'ground_waypoints': ground_waypoints,
                'aerial_robots_used': aerial_robots_used,
                'ground_robots_used': ground_robots_used,
                'num_vars': solution.get('num_variables', None),
                'num_constraints': solution.get('num_constraints', None)
            }
        else:
            metrics = {
                'map_size': map_size,
                'num_points': num_points,
                'solve_time': solve_time,
                'status': status,
                'objective_value': None,
                'aerial_waypoints': None,
                'ground_waypoints': None,
                'aerial_robots_used': None,
                'ground_robots_used': None,
                'num_vars': solution.get('num_variables', None),
                'num_constraints': solution.get('num_constraints', None)
            }
    except Exception as e:
        solve_time = time.time() - start_time
        
        metrics = {
            'map_size': map_size,
            'num_points': num_points,
            'solve_time': solve_time,
            'status': f"error: {str(e)}",
            'objective_value': None,
            'aerial_waypoints': None,
            'ground_waypoints': None,
            'aerial_robots_used': None,
            'ground_robots_used': None,
            'num_vars': None,
            'num_constraints': None
        }
    
    print(f"  Status: {metrics['status']}")
    print(f"  Solve time: {metrics['solve_time']:.2f} seconds")
    
    return metrics

def performance_experiment_map_size(
        base_params, 
        map_sizes=[100, 200, 500, 1000, 2000], 
        fixed_points=30,
        output_dir="performance_results"):
    """
    Experiment with increasing map size while keeping waypoint count fixed.
    """
    print(f"Testing performance with increasing map size (fixed {fixed_points} waypoints)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for size in map_sizes:
        metrics = measure_performance(size, fixed_points, base_params)
        results.append(metrics)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Save raw data
    df.to_csv(f"{output_dir}/map_size_scaling.csv", index=False)
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    plt.plot(df['map_size'], df['solve_time'], 'o-')
    plt.xlabel('Map Size')
    plt.ylabel('Solve Time (seconds)')
    plt.title('Solution Time vs Map Size')
    plt.grid(True)
    plt.savefig(f"{output_dir}/map_size_vs_time.png")
    
    return df

def performance_experiment_waypoints(
        base_params, 
        fixed_map_size=1000, 
        waypoint_counts=[10, 20, 30, 40, 50],
        output_dir="performance_results"):
    """
    Experiment with increasing waypoint count while keeping map size fixed.
    """
    print(f"Testing performance with increasing waypoint count (fixed map size {fixed_map_size})")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for num_points in waypoint_counts:
        metrics = measure_performance(fixed_map_size, num_points, base_params)
        results.append(metrics)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Save raw data
    df.to_csv(f"{output_dir}/waypoint_scaling.csv", index=False)
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    plt.plot(df['num_points'], df['solve_time'], 'o-')
    plt.xlabel('Number of Waypoints')
    plt.ylabel('Solve Time (seconds)')
    plt.title('Solution Time vs Number of Waypoints')
    plt.grid(True)
    plt.savefig(f"{output_dir}/waypoints_vs_time.png")
    
    # If we have optimal solutions, plot objective values
    if 'objective_value' in df and df['objective_value'].notna().any():
        plt.figure(figsize=(10, 6))
        plt.plot(df.loc[df['objective_value'].notna(), 'num_points'], 
                 df.loc[df['objective_value'].notna(), 'objective_value'], 'o-')
        plt.xlabel('Number of Waypoints')
        plt.ylabel('Objective Value')
        plt.title('Objective Value vs Number of Waypoints')
        plt.grid(True)
        plt.savefig(f"{output_dir}/waypoints_vs_objective.png")
    
    # If we have variable and constraint counts, plot model size
    if ('num_vars' in df and df['num_vars'].notna().any() and 
        'num_constraints' in df and df['num_constraints'].notna().any()):
        plt.figure(figsize=(10, 6))
        plt.plot(df['num_points'], df['num_vars'], 'o-', label='Variables')
        plt.plot(df['num_points'], df['num_constraints'], 's-', label='Constraints')
        plt.xlabel('Number of Waypoints')
        plt.ylabel('Count')
        plt.title('Model Size vs Number of Waypoints')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/waypoints_vs_model_size.png")
    
    return df

def analyze_computational_complexity(map_size_df, waypoint_df, output_dir="performance_results"):
    """
    Analyze the computational complexity by fitting curves to the performance data.
    """
    print("Analyzing computational complexity...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out non-optimal solutions
    map_size_df = map_size_df[map_size_df['status'] == 'optimal'].copy()
    waypoint_df = waypoint_df[waypoint_df['status'] == 'optimal'].copy()
    
    # Function to fit polynomial and calculate R^2
    def fit_and_plot(x, y, xlabel, ylabel, title, filename, degrees=[1, 2, 3]):
        plt.figure(figsize=(12, 8))
        plt.scatter(x, y, color='blue', label='Data points')
        
        colors = ['red', 'green', 'purple', 'orange']
        
        best_r2 = -1
        best_degree = 0
        
        for i, degree in enumerate(degrees):
            if len(x) <= degree:
                print(f"Warning: Not enough data points to fit degree {degree} polynomial")
                continue
                
            # Fit polynomial
            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            
            # Calculate R^2
            y_pred = poly(x)
            ss_total = np.sum((y - np.mean(y))**2)
            ss_residual = np.sum((y - y_pred)**2)
            r2 = 1 - (ss_residual / ss_total)
            
            # Update best fit
            if r2 > best_r2:
                best_r2 = r2
                best_degree = degree
            
            # Plot fitted curve
            x_line = np.linspace(min(x), max(x), 100)
            plt.plot(x_line, poly(x_line), color=colors[i % len(colors)], 
                     label=f'Degree {degree} polynomial (R² = {r2:.4f})')
            
            # Print equation
            eq = "y = "
            for j, coef in enumerate(coeffs):
                power = degree - j
                if power == 0:
                    eq += f"{coef:.4f}"
                elif power == 1:
                    eq += f"{coef:.4f}x + "
                else:
                    eq += f"{coef:.4f}x^{power} + "
            print(f"{title} - Degree {degree}: {eq} (R² = {r2:.4f})")
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title}\nBest fit: Degree {best_degree} (R² = {best_r2:.4f})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{filename}")
        
        return best_degree, best_r2
    
    results = {}
    
    # Analyze map size scaling
    if not map_size_df.empty and len(map_size_df) > 1:
        map_degree, map_r2 = fit_and_plot(
            map_size_df['map_size'], 
            map_size_df['solve_time'],
            'Map Size', 
            'Solve Time (seconds)',
            'Computational Complexity: Map Size vs. Solve Time',
            'complexity_map_size.png'
        )
        results['map_size_complexity'] = {
            'best_degree': map_degree,
            'r2': map_r2,
            'plot': 'complexity_map_size.png'
        }
    else:
        print("Not enough data points for map size complexity analysis")
        results['map_size_complexity'] = None
    
    # Analyze waypoint scaling
    if not waypoint_df.empty and len(waypoint_df) > 1:
        waypoint_degree, waypoint_r2 = fit_and_plot(
            waypoint_df['num_points'], 
            waypoint_df['solve_time'],
            'Number of Waypoints', 
            'Solve Time (seconds)',
            'Computational Complexity: Number of Waypoints vs. Solve Time',
            'complexity_waypoints.png'
        )
        results['waypoint_complexity'] = {
            'best_degree': waypoint_degree,
            'r2': waypoint_r2,
            'plot': 'complexity_waypoints.png'
        }
    else:
        print("Not enough data points for waypoint complexity analysis")
        results['waypoint_complexity'] = None
    
    return results

def analyze_solver_statistics(map_size_df, waypoint_df, output_dir="performance_results"):
    """
    Analyze solver statistics to understand performance bottlenecks
    """
    print("Analyzing solver statistics...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot solver time vs nodes explored (if available)
    if 'nodes_explored' in waypoint_df.columns and waypoint_df['nodes_explored'].notna().any():
        plt.figure(figsize=(10, 6))
        plt.scatter(waypoint_df['nodes_explored'], waypoint_df['solve_time'])
        plt.xlabel('Number of Nodes Explored')
        plt.ylabel('Solution Time (seconds)')
        plt.title('Solution Time vs Nodes Explored')
        plt.grid(True)
        plt.savefig(f"{output_dir}/time_vs_nodes.png")
    
    # Create a correlation heatmap for waypoint experiment metrics
    corr_cols = ['num_points', 'solve_time', 'objective_value']
    if 'num_vars' in waypoint_df.columns:
        corr_cols.append('num_vars')
    if 'num_constraints' in waypoint_df.columns:
        corr_cols.append('num_constraints')
    
    if len(corr_cols) > 2 and waypoint_df[corr_cols].notna().all().any():
        try:
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            corr_df = waypoint_df[corr_cols].corr()
            sns.heatmap(corr_df, annot=True, cmap='coolwarm')
            plt.title('Correlation Between Performance Metrics')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_heatmap.png")
        except ImportError:
            print("seaborn not available, skipping correlation heatmap")

def comprehensive_performance_analysis(output_dir="performance_results"):
    """
    Perform a comprehensive performance analysis on the MILP solver.
    """
    # Base parameters for testing
    base_params = {
        'aerial_speed': 50.0,
        'ground_speed': 20.0,
        'aerial_max_time': 60.0,
        'ground_max_time': 120.0,
        'aerial_inspection_time': 1.0,
        'ground_inspection_time': 3.0,
        'num_aerial_robots': 3,
        'num_ground_robots': 3
    }
    
    print("Starting Comprehensive Performance Analysis")
    print("===========================================")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run map size scaling experiment
    # Start with smaller sizes to ensure experiment completes
    map_size_df = performance_experiment_map_size(
        base_params, 
        map_sizes=[100, 200, 500, 1000],
        fixed_points=20,
        output_dir=output_dir
    )
    
    # Run waypoint scaling experiment
    # Start with fewer waypoints to ensure experiment completes
    waypoint_df = performance_experiment_waypoints(
        base_params, 
        fixed_map_size=500,
        waypoint_counts=[5, 10, 15, 20, 25],
        output_dir=output_dir
    )
    
    # Analyze computational complexity
    complexity_results = analyze_computational_complexity(
        map_size_df, 
        waypoint_df,
        output_dir=output_dir
    )
    
    # Analyze solver statistics
    analyze_solver_statistics(
        map_size_df,
        waypoint_df,
        output_dir=output_dir
    )
    
    # Generate summary report
    summary = []
    summary.append("# MILP Solver Performance Analysis Report")
    summary.append("\n## Overview")
    summary.append("This report presents the computational performance analysis of the Multi-Robot Inspection MILP solver")
    summary.append("as the problem scales in terms of map size and number of waypoints.")
    
    summary.append("\n## Map Size Scaling")
    summary.append(f"- Tested map sizes: {list(map_size_df['map_size'])}")
    summary.append(f"- Fixed number of waypoints: 20")
    summary.append("- See figures: map_size_vs_time.png")
    
    summary.append("\n## Waypoint Scaling")
    summary.append(f"- Fixed map size: 500 x 500")
    summary.append(f"- Tested waypoint counts: {list(waypoint_df['num_points'])}")
    summary.append("- See figures: waypoints_vs_time.png, waypoints_vs_objective.png")
    
    summary.append("\n## Computational Complexity Analysis")
    if complexity_results.get('map_size_complexity'):
        map_degree = complexity_results['map_size_complexity']['best_degree']
        map_r2 = complexity_results['map_size_complexity']['r2']
        summary.append(f"- Map size complexity: Polynomial of degree {map_degree} (R² = {map_r2:.4f})")
    
    if complexity_results.get('waypoint_complexity'):
        waypoint_degree = complexity_results['waypoint_complexity']['best_degree']
        waypoint_r2 = complexity_results['waypoint_complexity']['r2']
        summary.append(f"- Waypoint complexity: Polynomial of degree {waypoint_degree} (R² = {waypoint_r2:.4f})")
    
    summary.append("\n## Performance Summary")
    
    # Map size summary
    optimal_map_df = map_size_df[map_size_df['status'] == 'optimal']
    if not optimal_map_df.empty:
        min_map_time = optimal_map_df['solve_time'].min()
        max_map_time = optimal_map_df['solve_time'].max()
        min_map_size = optimal_map_df.loc[optimal_map_df['solve_time'].idxmin(), 'map_size']
        max_map_size = optimal_map_df.loc[optimal_map_df['solve_time'].idxmax(), 'map_size']
        
        summary.append(f"- Minimum solve time for map size experiment: {min_map_time:.2f} seconds (map size: {min_map_size})")
        summary.append(f"- Maximum solve time for map size experiment: {max_map_time:.2f} seconds (map size: {max_map_size})")
    
    # Waypoint summary
    optimal_waypoint_df = waypoint_df[waypoint_df['status'] == 'optimal']
    if not optimal_waypoint_df.empty:
        min_waypoint_time = optimal_waypoint_df['solve_time'].min()
        max_waypoint_time = optimal_waypoint_df['solve_time'].max()
        min_waypoints = optimal_waypoint_df.loc[optimal_waypoint_df['solve_time'].idxmin(), 'num_points']
        max_waypoints = optimal_waypoint_df.loc[optimal_waypoint_df['solve_time'].idxmax(), 'num_points']
        
        summary.append(f"- Minimum solve time for waypoint experiment: {min_waypoint_time:.2f} seconds (waypoints: {min_waypoints})")
        summary.append(f"- Maximum solve time for waypoint experiment: {max_waypoint_time:.2f} seconds (waypoints: {max_waypoints})")
    
    # Non-optimal solutions
    non_optimal_count = len(map_size_df[map_size_df['status'] != 'optimal']) + len(waypoint_df[waypoint_df['status'] != 'optimal'])
    if non_optimal_count > 0:
        summary.append(f"\n- Number of non-optimal solutions: {non_optimal_count}")
    
    summary.append("\n## Conclusions")
    summary.append("The performance analysis reveals how the MILP solver scales with both map size and number of waypoints.")
    summary.append("The polynomial fits to the solve time data give insight into the computational complexity class of the problem.")
    summary.append("These results can be used to predict performance for larger problem instances and to identify potential")
    summary.append("bottlenecks in the solution approach.")
    
    summary_text = "\n".join(summary)
    
    with open(f"{output_dir}/performance_summary.md", "w") as f:
        f.write(summary_text)
    
    print("\nPerformance analysis complete. Results saved to:", output_dir)
    
    return {
        'map_size_df': map_size_df,
        'waypoint_df': waypoint_df,
        'complexity_results': complexity_results,
        'summary_report': f"{output_dir}/performance_summary.md"
    }

if __name__ == "__main__":
    comprehensive_performance_analysis()