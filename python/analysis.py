# analysis.py
"""
Sensitivity analysis for multi-robot inspection problem using direct perturbation method
"""

from solver import solve_multi_robot_inspection_problem

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Base parameters - adjusted for large map
base_params = {
    'aerial_speed': 50.0,       # Increased for large map (m/min)
    'ground_speed': 20.0,       # Increased for large map (m/min)
    'aerial_max_time': 60.0,    # Increased for large map (min)
    'ground_max_time': 120.0,   # Increased for large map (min)
    'aerial_inspection_time': 1.0,  # Same as before (min)
    'ground_inspection_time': 3.0,  # Same as before (min)
    'num_aerial_robots': 3,
    'num_ground_robots': 3
}

def generate_large_scale_waypoints(map_size=1000, num_points=30):
    """
    Generate waypoints for a large-scale map with points distributed in relation to the diagonal.
    
    Args:
        map_size: Size of the map (square, map_size x map_size)
        num_points: Approximate number of waypoints to generate
        
    Returns:
        waypoints: List of waypoint coordinates
        aerial_depot: Coordinates of aerial depot
        ground_depot: Coordinates of ground depot
    """
    # Place depots on opposite sides of the diagonal
    aerial_depot = (0, 0)  # Bottom left
    ground_depot = (map_size, map_size)  # Top right
    
    # Number of points in each region (left of diagonal, on diagonal, right of diagonal)
    points_per_region = num_points // 3
    
    # Generate points along the diagonal
    diagonal_points = []
    for i in range(points_per_region):
        # Evenly space points along diagonal
        t = (i + 1) / (points_per_region + 1)
        x = t * map_size
        y = t * map_size
        diagonal_points.append((x, y))
    
    # Generate points to the left of diagonal
    left_points = []
    for i in range(points_per_region):
        # Create points at increasing distances from diagonal
        t = (i + 1) / (points_per_region + 1)
        x = t * map_size
        # Shift down from the diagonal by varying amounts
        y = t * map_size - (t * 300)  # Larger offset for points further along diagonal
        if y < 0:
            y = 50  # Ensure point is within map
        left_points.append((x, y))
    
    # Generate points to the right of diagonal
    right_points = []
    for i in range(points_per_region):
        # Create points at increasing distances from diagonal
        t = (i + 1) / (points_per_region + 1)
        # Shift left from the diagonal by varying amounts
        x = t * map_size - (t * 300)  # Larger offset for points further along diagonal
        if x < 0:
            x = 50  # Ensure point is within map
        y = t * map_size
        right_points.append((x, y))
    
    # Combine all points
    waypoints = diagonal_points + left_points + right_points
    
    return waypoints, aerial_depot, ground_depot


def sensitivity_analysis():
    """
    Perform sensitivity analysis on a much larger map (1000x1000m) to better stress test parameters.
    """
    

    
    # Generate large-scale waypoints (1000x1000 map)
    map_size = 1000
    waypoints, aerial_depot, ground_depot = generate_large_scale_waypoints(map_size=map_size, num_points=30)
    
    
    # Parameters to vary and their ranges (scaled for large map)
    param_ranges = {
        'aerial_speed': [25.0, 50.0, 75.0, 100.0],
        'ground_speed': [10.0, 20.0, 30.0, 40.0],
        'aerial_max_time': [30.0, 60.0, 90.0, 120.0],
        'ground_max_time': [60.0, 120.0, 180.0, 240.0],
        'aerial_inspection_time': [0.5, 1.0, 1.5, 2.0],
        'ground_inspection_time': [1.5, 3.0, 4.5, 6.0],
        'num_aerial_robots': [1, 3, 5, 7],
        'num_ground_robots': [1, 3, 5, 7]
    }
    
    # Results storage
    results = []
    
    # Perform one-at-a-time sensitivity analysis
    for param_name, param_values in param_ranges.items():
        for value in param_values:
            # Skip base case to avoid duplicates
            if value == base_params[param_name]:
                continue
                
            # Create modified params
            current_params = base_params.copy()
            current_params[param_name] = value
            
            # Run the model with modified params
            print(f"Running with {param_name} = {value}...")
            solution = solve_multi_robot_inspection_problem(
                waypoints, aerial_depot, ground_depot,
                current_params['aerial_speed'], 
                current_params['ground_speed'],
                current_params['aerial_max_time'], 
                current_params['ground_max_time'],
                current_params['aerial_inspection_time'], 
                current_params['ground_inspection_time'],
                current_params['num_aerial_robots'], 
                current_params['num_ground_robots']
            )
            
            # Extract metrics of interest
            if solution["status"] == "optimal":
                metrics = {
                    'param_name': param_name,
                    'param_value': value,
                    'objective_value': solution['objective_value'],
                    'aerial_waypoints': sum(sum(robot.values()) for robot in solution['aerial_visited'].values()),
                    'ground_waypoints': sum(sum(robot.values()) for robot in solution['ground_visited'].values()),
                    'aerial_robots_used': sum(1 for k, visited in solution['aerial_visited'].items() 
                                           if sum(visited.values()) > 0),
                    'ground_robots_used': sum(1 for l, visited in solution['ground_visited'].items() 
                                           if sum(visited.values()) > 0),
                    'max_aerial_time': max([solution['total_aerial_times'][k] 
                                          for k in solution['total_aerial_times']] or [0]),
                    'max_ground_time': max([solution['total_ground_times'][l] 
                                          for l in solution['total_ground_times']] or [0])
                }
                results.append(metrics)
            else:
                print(f"No feasible solution for {param_name} = {value}")
    
    # Run base case
    print("Running base case...")
    base_solution = solve_multi_robot_inspection_problem(
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
    
    if base_solution["status"] == "optimal":
        base_metrics = {
            'param_name': 'base_case',
            'param_value': 'base',
            'objective_value': base_solution['objective_value'],
            'aerial_waypoints': sum(sum(robot.values()) for robot in base_solution['aerial_visited'].values()),
            'ground_waypoints': sum(sum(robot.values()) for robot in base_solution['ground_visited'].values()),
            'aerial_robots_used': sum(1 for k, visited in base_solution['aerial_visited'].items() 
                                   if sum(visited.values()) > 0),
            'ground_robots_used': sum(1 for l, visited in base_solution['ground_visited'].items() 
                                   if sum(visited.values()) > 0),
            'max_aerial_time': max([base_solution['total_aerial_times'][k] 
                                  for k in base_solution['total_aerial_times']] or [0]),
            'max_ground_time': max([base_solution['total_ground_times'][l] 
                                  for l in base_solution['total_ground_times']] or [0])
        }
        results.append(base_metrics)
    
    
    return results


def visualize_sensitivity_analysis(results, output_dir="analysis_results"):
    """
    Visualize sensitivity analysis results.
    """
    # Create output directory if it doesn't exist
    if output_dir == None:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Get base case values
    base_case = df[df['param_name'] == 'base_case'].iloc[0]
    base_obj = base_case['objective_value']
    
    # Calculate percentage change from base case
    df['obj_pct_change'] = (df['objective_value'] - base_obj) / base_obj * 100
    
    # Create parameter-specific DataFrames for plotting
    param_dfs = {}

    
    for param in df['param_name'].unique():
        if param != 'base_case':
            param_dfs[param] = df[df['param_name'] == param].copy()
            
            # Add base case for reference
            base_row = df[df['param_name'] == 'base_case'].copy()
            base_row['param_value'] = base_params[param]  # Fixed: Use base_params instead
            param_dfs[param] = pd.concat([param_dfs[param], base_row])
            
            # Sort by parameter value
            try:
                param_dfs[param]['param_value'] = pd.to_numeric(param_dfs[param]['param_value'])
                param_dfs[param] = param_dfs[param].sort_values('param_value')
            except:
                pass
    
    # Create tornado chart for overall sensitivity
    plt.figure(figsize=(10, 8))
    
    # Get max absolute percentage change for each parameter
    sensitivity = {}
    for param, param_df in param_dfs.items():
        param_df_no_base = param_df[param_df['param_name'] != 'base_case']
        if not param_df_no_base.empty:
            sensitivity[param] = param_df_no_base['obj_pct_change'].abs().max()
    
    # Create tornado chart
    param_labels = list(sensitivity.keys())
    param_values = list(sensitivity.values())
    
    # Sort by sensitivity
    sorted_indices = np.argsort(param_values)[::-1]  # Reverse to get descending order
    sorted_params = [param_labels[i] for i in sorted_indices]
    sorted_values = [param_values[i] for i in sorted_indices]
    
    plt.barh(sorted_params, sorted_values)
    plt.xlabel('Maximum % Change in Objective Value')
    plt.title('Parameter Sensitivity - Tornado Chart')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tornado_chart.png')
    
    # Create detailed parameter plots
    for param, param_df in param_dfs.items():
        plt.figure(figsize=(10, 6))
        
        # Filter out base case for plotting
        param_df_no_base = param_df[param_df['param_name'] != 'base_case']
        plt.plot(param_df_no_base['param_value'], param_df_no_base['objective_value'], 'o-', label='Objective Value')
        
        # Add vertical line for base value
        base_value = base_params[param]
        plt.axvline(x=base_value, color='r', linestyle='--', label=f'Base Value: {base_value}')
        
        plt.xlabel(f'{param}')
        plt.ylabel('Objective Value')
        plt.title(f'Sensitivity to {param}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sensitivity_{param}.png')
    
    # Create correlation heatmap for all metrics
    try:
        import seaborn as sns
        plt.figure(figsize=(12, 10))
        # Remove categorical columns and filter out base case
        numeric_df = df[df['param_name'] != 'base_case'].select_dtypes(include=['number'])
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png')
    except ImportError:
        print("Seaborn not available, skipping correlation heatmap")
    
    # Create elasticity table
    elasticity = {}
    for param, param_df in param_dfs.items():
        if param == 'base_case':
            continue
            
        # Get base values
        base_param_value = base_params[param]
        base_obj_value = base_obj
        
        # Calculate elasticity for each point
        param_elasticity = []
        for _, row in param_df[param_df['param_name'] == param].iterrows():
            param_value = float(row['param_value'])
            obj_value = float(row['objective_value'])
            
            # Skip base case
            if param_value == base_param_value:
                continue
                
            # Calculate percentage changes
            param_pct_change = (param_value - base_param_value) / base_param_value
            obj_pct_change = (obj_value - base_obj_value) / base_obj_value
            
            # Calculate elasticity
            if abs(param_pct_change) > 1e-6:
                elasticity_value = obj_pct_change / param_pct_change
                param_elasticity.append((param_value, elasticity_value))
        
        elasticity[param] = param_elasticity
    
    # Create elasticity table
    elasticity_data = []
    for param, values in elasticity.items():
        for param_value, elasticity_value in values:
            elasticity_data.append({
                'Parameter': param,
                'Parameter Value': param_value,
                'Elasticity': elasticity_value,
                '% Change from Base': (param_value - base_params[param]) / base_params[param] * 100
            })
    
    elasticity_df = pd.DataFrame(elasticity_data)
    print("Elasticity Table:")
    print(elasticity_df)
    
    # Save elasticity data
    elasticity_df.to_csv(f'{output_dir}/elasticity.csv', index=False)
    
    return {
        'tornado_chart': 'tornado_chart.png',
        'parameter_plots': [f'sensitivity_{param}.png' for param in param_dfs.keys()],
        'correlation_heatmap': 'correlation_heatmap.png',
        'elasticity_table': elasticity_df
    }




if __name__ == "__main__":
    results = sensitivity_analysis()
    visualize_sensitivity_analysis(results)