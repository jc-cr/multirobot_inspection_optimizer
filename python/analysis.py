import pulp
import numpy as np
import pandas as pd
from solver import solve_multi_robot_inspection_problem

def perform_sensitivity_analysis(
        waypoints, 
        aerial_depot, 
        ground_depot,
        aerial_speed,
        ground_speed,
        aerial_max_time,
        ground_max_time,
        aerial_inspection_time, 
        ground_inspection_time,
        num_aerial_robots,
        num_ground_robots,
        parameters_to_analyze=None):
    """
    Perform sensitivity analysis on a multi-robot inspection MILP problem using a two-phase approach.
    
    Parameters:
    - All the standard parameters for the solve_multi_robot_inspection_problem function
    - parameters_to_analyze: dict of parameters to analyze with their ranges, e.g.,
      {'aerial_speed': [3.0, 4.0, 5.0, 6.0, 7.0], 'ground_speed': [1.0, 1.5, 2.0, 2.5, 3.0]}
      
    Returns:
    - Dictionary containing sensitivity analysis results
    """
    # Step 1: Solve the original MILP problem
    print("Solving original MILP problem...")
    base_solution = solve_multi_robot_inspection_problem(
        waypoints, aerial_depot, ground_depot,
        aerial_speed, ground_speed,
        aerial_max_time, ground_max_time,
        aerial_inspection_time, ground_inspection_time,
        num_aerial_robots, num_ground_robots
    )
    
    if base_solution["status"] != "optimal":
        return {"error": "Original problem could not be solved optimally"}
    
    # Step 2: Extract the optimal integer variable values
    n = len(waypoints)
    k_max = num_aerial_robots
    l_max = num_ground_robots
    
    # Create mappings for integer variables (aerial and ground robot assignments)
    aerial_assignments = {}
    for k in range(k_max):
        for i in range(n):
            if base_solution["aerial_visited"][k].get(i, 0) > 0.5:
                aerial_assignments[(i, k)] = 1
            else:
                aerial_assignments[(i, k)] = 0
    
    ground_assignments = {}
    for l in range(l_max):
        for i in range(n):
            if base_solution["ground_visited"][l].get(i, 0) > 0.5:
                ground_assignments[(i, l)] = 1
            else:
                ground_assignments[(i, l)] = 0
    
    # Step 3: Create a new LP model with integer variables fixed
    def create_lp_with_fixed_integers():
        """Create and solve LP model with integer variables fixed at optimal values"""
        # Number of waypoints and robots
        aerial_depot_idx = -1
        ground_depot_idx = -2
        
        # Calculate distances
        distances = {}
        
        # Calculate waypoint-to-waypoint distances
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt((waypoints[i][0] - waypoints[j][0])**2 + 
                                  (waypoints[i][1] - waypoints[j][1])**2)
                    distances[(i, j)] = dist
        
        # Calculate depot-to-waypoint distances
        for i in range(n):
            distances[(aerial_depot_idx, i)] = np.sqrt((aerial_depot[0] - waypoints[i][0])**2 + 
                                                     (aerial_depot[1] - waypoints[i][1])**2)
            distances[(i, aerial_depot_idx)] = distances[(aerial_depot_idx, i)]
            
            distances[(ground_depot_idx, i)] = np.sqrt((ground_depot[0] - waypoints[i][0])**2 + 
                                                     (ground_depot[1] - waypoints[i][1])**2)
            distances[(i, ground_depot_idx)] = distances[(ground_depot_idx, i)]
        
        # Calculate travel times based on distances and speeds
        aerial_travel_times = {k_v: v / aerial_speed for k_v, v in distances.items()}
        ground_travel_times = {k_v: v / ground_speed for k_v, v in distances.items()}
        
        # Create the LP model (relaxed version of the MILP)
        model = pulp.LpProblem("MultiRobotInspection_LP", pulp.LpMaximize)
        
        # Calculate appropriate Big-M values
        max_aerial_travel = max(aerial_travel_times.values())
        M_A = aerial_max_time + max_aerial_travel + aerial_inspection_time
        
        max_ground_travel = max(ground_travel_times.values())
        M_G = ground_max_time + max_ground_travel + ground_inspection_time
        
        # Sets for indexing
        N = range(n)  # Waypoints
        K = range(k_max)  # Aerial robots
        L = range(l_max)  # Ground robots
        
        # Decision Variables - but we'll fix the binary ones
        # Time when aerial robot k completes inspection at waypoint i
        a_time = {(i, k): pulp.LpVariable(f"a_time_{i}_{k}", lowBound=0) 
                  for i in N for k in K if aerial_assignments.get((i, k), 0) > 0.5}
        
        # Time when ground robot l completes inspection at waypoint i
        g_time = {(i, l): pulp.LpVariable(f"g_time_{i}_{l}", lowBound=0) 
                  for i in N for l in L if ground_assignments.get((i, l), 0) > 0.5}
        
        # Binary variables to indicate if a robot is used - fixed to their values
        use_aerial = {k: aerial_assignments.get((i, k), 0) > 0.5 for k in K for i in N}
        use_aerial = {k: 1 if any(use_aerial[k] for i in N) else 0 for k in K}
        
        use_ground = {l: ground_assignments.get((i, l), 0) > 0.5 for l in L for i in N}
        use_ground = {l: 1 if any(use_ground[l] for i in N) else 0 for l in L}
        
        # Objective: Maximize the number of waypoints visited by ground robots
        # This is now fixed since we've fixed the binary variables, but we include it for sensitivity analysis
        model += pulp.lpSum(ground_assignments.get((i, l), 0) for i in N for l in L)
        
        # Constraints - we only include constraints relevant to continuous variables
        # Time constraints for aerial robots
        for k in K:
            if use_aerial[k] > 0.5:
                for i in N:
                    if aerial_assignments.get((i, k), 0) > 0.5:
                        # If robot k visits waypoint i, it must spend at least inspection time there
                        model += a_time[(i, k)] >= aerial_inspection_time
                        
                        # Total time must be within max operation time
                        model += a_time[(i, k)] + aerial_travel_times[(i, aerial_depot_idx)] <= aerial_max_time
        
        # Time constraints for ground robots
        for l in L:
            if use_ground[l] > 0.5:
                for i in N:
                    if ground_assignments.get((i, l), 0) > 0.5:
                        # If robot l visits waypoint i, it must spend at least inspection time there
                        model += g_time[(i, l)] >= ground_inspection_time
                        
                        # Total time must be within max operation time
                        model += g_time[(i, l)] + ground_travel_times[(i, ground_depot_idx)] <= ground_max_time
        
        # Precedence constraints: ground robot can only inspect after aerial robot has inspected
        for i in N:
            for k in K:
                for l in L:
                    if (aerial_assignments.get((i, k), 0) > 0.5 and 
                        ground_assignments.get((i, l), 0) > 0.5 and
                        use_aerial[k] > 0.5 and
                        use_ground[l] > 0.5):
                        # Ground robot visits after aerial robot inspection
                        model += g_time[(i, l)] >= a_time[(i, k)]
        
        # Approximate route length constraints for aerial robots
        for k in K:
            if use_aerial[k] > 0.5:
                visited_waypoints = [i for i in N if aerial_assignments.get((i, k), 0) > 0.5]
                if visited_waypoints:
                    # Route constraint
                    total_travel_time = sum(2 * aerial_travel_times[(aerial_depot_idx, i)] for i in visited_waypoints)
                    total_inspection_time = sum(aerial_inspection_time for i in visited_waypoints)
                    
                    # Track total time constraint
                    model += pulp.lpSum(a_time[(i, k)] for i in visited_waypoints) + total_travel_time <= aerial_max_time
        
        # Approximate route length constraints for ground robots
        for l in L:
            if use_ground[l] > 0.5:
                visited_waypoints = [i for i in N if ground_assignments.get((i, l), 0) > 0.5]
                if visited_waypoints:
                    # Route constraint
                    total_travel_time = sum(2 * ground_travel_times[(ground_depot_idx, i)] for i in visited_waypoints)
                    total_inspection_time = sum(ground_inspection_time for i in visited_waypoints)
                    
                    # Track total time constraint
                    model += pulp.lpSum(g_time[(i, l)] for i in visited_waypoints) + total_travel_time <= ground_max_time
        
        # Solve the LP model
        print("Solving the LP relaxation with fixed integer variables...")
        solver = pulp.PULP_CBC_CMD(msg=True)
        model.solve(solver)
        
        # Extract sensitivity information
        sensitivity_info = {
            "status": pulp.LpStatus[model.status],
            "objective_value": pulp.value(model.objective),
            "constraints": {},
            "variables": {}
        }
        
        # Get shadow prices and slack values for constraints
        for name, constraint in model.constraints.items():
            sensitivity_info["constraints"][name] = {
                "shadow_price": constraint.pi,
                "slack": constraint.slack
            }
        
        # Get reduced costs for variables
        for var in model.variables():
            sensitivity_info["variables"][var.name] = {
                "value": var.value(),
                "reduced_cost": var.dj
            }
        
        return sensitivity_info, model
    
    # Perform base LP analysis
    lp_sensitivity, lp_model = create_lp_with_fixed_integers()
    
    # Step 4: Perform sensitivity analysis on parameters of interest
    sensitivity_results = {
        "base_solution": base_solution,
        "lp_sensitivity": lp_sensitivity,
        "parameter_analysis": {}
    }
    
    # If specific parameters are provided for analysis
    if parameters_to_analyze:
        for param_name, param_values in parameters_to_analyze.items():
            print(f"Analyzing sensitivity to {param_name}...")
            param_results = []
            
            # Store original parameter value
            original_value = locals()[param_name]
            
            for param_value in param_values:
                # Create parameter set with this value
                current_params = {
                    "aerial_speed": aerial_speed,
                    "ground_speed": ground_speed,
                    "aerial_max_time": aerial_max_time,
                    "ground_max_time": ground_max_time,
                    "aerial_inspection_time": aerial_inspection_time,
                    "ground_inspection_time": ground_inspection_time,
                    "num_aerial_robots": num_aerial_robots,
                    "num_ground_robots": num_ground_robots
                }
                
                # Update the specific parameter
                current_params[param_name] = param_value
                
                # Solve with updated parameters
                modified_solution = solve_multi_robot_inspection_problem(
                    waypoints, aerial_depot, ground_depot,
                    current_params["aerial_speed"], 
                    current_params["ground_speed"],
                    current_params["aerial_max_time"], 
                    current_params["ground_max_time"],
                    current_params["aerial_inspection_time"], 
                    current_params["ground_inspection_time"],
                    current_params["num_aerial_robots"], 
                    current_params["num_ground_robots"]
                )
                
                # Record results
                param_results.append({
                    "param_value": param_value,
                    "objective_value": modified_solution.get("objective_value", None),
                    "status": modified_solution.get("status", None),
                    "solution_structure_changed": is_solution_structure_changed(base_solution, modified_solution)
                })
            
            # Restore original parameter value
            locals()[param_name] = original_value
            
            # Store analysis results
            sensitivity_results["parameter_analysis"][param_name] = param_results
            
            # Find allowable range where solution structure remains the same
            allowable_range = find_allowable_range(param_results, original_value)
            sensitivity_results["allowable_ranges"] = sensitivity_results.get("allowable_ranges", {})
            sensitivity_results["allowable_ranges"][param_name] = allowable_range
    
    return sensitivity_results

def is_solution_structure_changed(base_solution, new_solution):
    """Check if the solution structure has changed (different assignment of waypoints to robots)"""
    if base_solution.get("status") != new_solution.get("status"):
        return True
    
    # Check if the same waypoints are visited by the same robots
    for k in base_solution["aerial_visited"]:
        if k not in new_solution["aerial_visited"]:
            return True
        for i in base_solution["aerial_visited"][k]:
            if (base_solution["aerial_visited"][k].get(i, 0) > 0.5) != (new_solution["aerial_visited"][k].get(i, 0) > 0.5):
                return True
    
    for l in base_solution["ground_visited"]:
        if l not in new_solution["ground_visited"]:
            return True
        for i in base_solution["ground_visited"][l]:
            if (base_solution["ground_visited"][l].get(i, 0) > 0.5) != (new_solution["ground_visited"][l].get(i, 0) > 0.5):
                return True
    
    return False

def find_allowable_range(param_results, original_value):
    """Find the range of parameter values where solution structure remains the same"""
    # Sort by parameter value
    sorted_results = sorted(param_results, key=lambda x: x["param_value"])
    
    # Find range where solution structure is unchanged
    lower_bound = None
    upper_bound = None
    
    # Find the original value in the results
    original_idx = None
    for i, result in enumerate(sorted_results):
        if abs(result["param_value"] - original_value) < 1e-6:
            original_idx = i
            break
    
    if original_idx is None:
        return {"lower_bound": None, "upper_bound": None, "error": "Original value not found in results"}
    
    # Look for lower bound
    for i in range(original_idx, -1, -1):
        if sorted_results[i]["solution_structure_changed"]:
            lower_bound = sorted_results[i+1]["param_value"] if i+1 < len(sorted_results) else None
            break
        if i == 0:
            lower_bound = sorted_results[0]["param_value"]
    
    # Look for upper bound
    for i in range(original_idx, len(sorted_results)):
        if sorted_results[i]["solution_structure_changed"]:
            upper_bound = sorted_results[i-1]["param_value"] if i > 0 else None
            break
        if i == len(sorted_results) - 1:
            upper_bound = sorted_results[-1]["param_value"]
    
    return {"lower_bound": lower_bound, "upper_bound": upper_bound}

def analyze_multiple_parameters_simultaneously(
        waypoints, 
        aerial_depot, 
        ground_depot,
        base_params,
        param_combinations):
    """
    Analyze the effect of varying multiple parameters simultaneously.
    
    Parameters:
    - waypoints, aerial_depot, ground_depot: Problem geometry
    - base_params: Dictionary with base parameter values
    - param_combinations: List of dictionaries, each with parameter combinations to test
    
    Returns:
    - DataFrame with results of all combinations
    """
    results = []
    
    for combo in param_combinations:
        # Create a copy of base parameters and update with this combination
        params = base_params.copy()
        params.update(combo)
        
        # Solve the MILP with these parameters
        solution = solve_multi_robot_inspection_problem(
            waypoints, aerial_depot, ground_depot,
            params["aerial_speed"], params["ground_speed"],
            params["aerial_max_time"], params["ground_max_time"],
            params["aerial_inspection_time"], params["ground_inspection_time"],
            params["num_aerial_robots"], params["num_ground_robots"]
        )
        
        # Record results
        result = combo.copy()
        result["objective_value"] = solution.get("objective_value", None)
        result["status"] = solution.get("status", None)
        
        # Calculate utilization metrics
        aerial_robots_used = sum(1 for k in range(params["num_aerial_robots"]) 
                               if k in solution["aerial_visited"] and sum(solution["aerial_visited"][k].values()) > 0)
        ground_robots_used = sum(1 for l in range(params["num_ground_robots"]) 
                               if l in solution["ground_visited"] and sum(solution["ground_visited"][l].values()) > 0)
        
        result["aerial_robots_used"] = aerial_robots_used
        result["ground_robots_used"] = ground_robots_used
        result["aerial_utilization"] = aerial_robots_used / params["num_aerial_robots"] * 100 if params["num_aerial_robots"] > 0 else 0
        result["ground_utilization"] = ground_robots_used / params["num_ground_robots"] * 100 if params["num_ground_robots"] > 0 else 0
        
        results.append(result)
    
    return pd.DataFrame(results)

def visualize_sensitivity_results(sensitivity_results, output_file=None):
    """
    Visualize sensitivity analysis results.
    
    Parameters:
    - sensitivity_results: Output from perform_sensitivity_analysis
    - output_file: If provided, save the figures to this file
    
    Returns:
    - None (displays or saves figures)
    """
    import matplotlib.pyplot as plt
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot parameter analysis results
    subplot_count = len(sensitivity_results["parameter_analysis"])
    if subplot_count == 0:
        print("No parameter analysis results to visualize")
        return
    
    # Calculate rows and columns for subplots
    cols = min(3, subplot_count)
    rows = (subplot_count + cols - 1) // cols
    
    for i, (param_name, param_results) in enumerate(sensitivity_results["parameter_analysis"].items()):
        plt.subplot(rows, cols, i + 1)
        
        # Extract data for plotting
        param_values = [r["param_value"] for r in param_results]
        obj_values = [r["objective_value"] for r in param_results]
        structure_changed = [r["solution_structure_changed"] for r in param_results]
        
        # Plot objective values
        plt.plot(param_values, obj_values, 'b-o', label='Objective Value')
        
        # Highlight points where solution structure changes
        changed_x = [param_values[i] for i in range(len(param_values)) if structure_changed[i]]
        changed_y = [obj_values[i] for i in range(len(obj_values)) if structure_changed[i]]
        if changed_x:
            plt.scatter(changed_x, changed_y, color='red', s=100, label='Structure Changed')
        
        # If allowable ranges are available, mark them
        if "allowable_ranges" in sensitivity_results and param_name in sensitivity_results["allowable_ranges"]:
            range_info = sensitivity_results["allowable_ranges"][param_name]
            lower = range_info.get("lower_bound")
            upper = range_info.get("upper_bound")
            
            if lower is not None and upper is not None:
                plt.axvline(x=lower, color='g', linestyle='--', label='Allowable Range')
                plt.axvline(x=upper, color='g', linestyle='--')
                plt.axvspan(lower, upper, alpha=0.2, color='green')
        
        plt.title(f'Sensitivity to {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Objective Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create example data
    waypoints = [(1, 5), (3, 3), (5, 5), (7, 3), (9, 5), (5, 7), (2, 8), (8, 8), (3, 1), (7, 1)]
    aerial_depot = (0, 0)
    ground_depot = (10, 0)
    
    # Base parameters
    base_params = {
        "aerial_speed": 5.0,
        "ground_speed": 2.0,
        "aerial_max_time": 30.0,
        "ground_max_time": 60.0,
        "aerial_inspection_time": 1.0,
        "ground_inspection_time": 3.0,
        "num_aerial_robots": 2,
        "num_ground_robots": 2
    }
    
    # Parameters to analyze
    parameters_to_analyze = {
        "aerial_speed": [3.0, 4.0, 5.0, 6.0, 7.0],
        "ground_speed": [1.0, 1.5, 2.0, 2.5, 3.0],
        "aerial_max_time": [20.0, 25.0, 30.0, 35.0, 40.0],
        "ground_max_time": [40.0, 50.0, 60.0, 70.0, 80.0],
        "num_aerial_robots": [1, 2, 3, 4, 5],
        "num_ground_robots": [1, 2, 3, 4, 5]
    }
    
    # Perform sensitivity analysis
    sensitivity_results = perform_sensitivity_analysis(
        waypoints, aerial_depot, ground_depot,
        base_params["aerial_speed"], base_params["ground_speed"],
        base_params["aerial_max_time"], base_params["ground_max_time"],
        base_params["aerial_inspection_time"], base_params["ground_inspection_time"],
        base_params["num_aerial_robots"], base_params["num_ground_robots"],
        parameters_to_analyze
    )
    
    # Visualize results
    visualize_sensitivity_results(sensitivity_results, "analysis_results/sensitivity_analysis_results.png")
    
    # Use the same parameters from the single-parameter analysis
    # but create reasonable combinations to test together
    param_combinations = [
        # Test robot count combinations (keeping other parameters at baseline)
        {"num_aerial_robots": 1, "num_ground_robots": 1},
        {"num_aerial_robots": 1, "num_ground_robots": 3},
        {"num_aerial_robots": 3, "num_ground_robots": 1},
        {"num_aerial_robots": 3, "num_ground_robots": 3},
        
        # Test speed combinations
        {"aerial_speed": 3.0, "ground_speed": 1.0},
        {"aerial_speed": 3.0, "ground_speed": 3.0},
        {"aerial_speed": 7.0, "ground_speed": 1.0},
        {"aerial_speed": 7.0, "ground_speed": 3.0},
        
        # Test time combinations
        {"aerial_max_time": 20.0, "ground_max_time": 40.0},
        {"aerial_max_time": 20.0, "ground_max_time": 80.0},
        {"aerial_max_time": 40.0, "ground_max_time": 40.0},
        {"aerial_max_time": 40.0, "ground_max_time": 80.0},
        
        # Test mixed critical parameter combinations
        {"aerial_speed": 3.0, "num_aerial_robots": 1},
        {"aerial_speed": 7.0, "num_aerial_robots": 3},
        {"ground_speed": 1.0, "num_ground_robots": 1},
        {"ground_speed": 3.0, "num_ground_robots": 3}
    ]
        
    multi_param_results = analyze_multiple_parameters_simultaneously(
        waypoints, aerial_depot, ground_depot,
        base_params, param_combinations
    )
    
    print("Multi-parameter analysis results:")
    print(multi_param_results)