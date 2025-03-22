import pulp
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from matplotlib.lines import Line2D

def solve_robot_inspection_problem(waypoints, aerial_depot, ground_depot, 
                                 aerial_speed, ground_speed,
                                 aerial_max_time, ground_max_time, 
                                 aerial_inspection_time, ground_inspection_time):
    """
    Solves the robot inspection problem with different speeds for aerial and ground robots.
    
    Parameters:
    -----------
    waypoints : list of tuples (x, y)
        Coordinates of inspection waypoints
    aerial_depot : tuple (x, y)
        Coordinates of aerial robot depot
    ground_depot : tuple (x, y)
        Coordinates of ground robot depot
    aerial_speed : float
        Speed of aerial robot (distance units per minute)
    ground_speed : float
        Speed of ground robot (distance units per minute)
    aerial_max_time : float
        Maximum operation time for aerial robot (minutes)
    ground_max_time : float
        Maximum operation time for ground robot (minutes)
    aerial_inspection_time : float
        Time required for aerial robot to inspect a waypoint (minutes)
    ground_inspection_time : float
        Time required for ground robot to inspect a waypoint (minutes)
        
    Returns:
    --------
    dict : A dictionary containing the solution information
    """
    # Number of waypoints
    n = len(waypoints)
    
    # Add depot indices
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
    aerial_travel_times = {k: v / aerial_speed for k, v in distances.items()}
    ground_travel_times = {k: v / ground_speed for k, v in distances.items()}
    
    # Create the optimization model
    model = pulp.LpProblem("RobotInspection", pulp.LpMaximize)
    
    # Big M constant
    M = 1000
    
    # Sets for indexing
    N = range(n)  # Waypoints
    
    # Create decision variables
    
    # Route variables
    x_a = {}
    for i in list(N) + [aerial_depot_idx]:
        for j in list(N) + [aerial_depot_idx]:
            if i != j:
                x_a[(i, j)] = pulp.LpVariable(f"x_a_{i}_{j}", cat=pulp.LpBinary)
    
    x_g = {}
    for i in list(N) + [ground_depot_idx]:
        for j in list(N) + [ground_depot_idx]:
            if i != j:
                x_g[(i, j)] = pulp.LpVariable(f"x_g_{i}_{j}", cat=pulp.LpBinary)
    
    # Visit variables
    v_a = {i: pulp.LpVariable(f"v_a_{i}", cat=pulp.LpBinary) for i in N}
    v_g = {i: pulp.LpVariable(f"v_g_{i}", cat=pulp.LpBinary) for i in N}
    
    # Position variables for subtour elimination
    u_a = {i: pulp.LpVariable(f"u_a_{i}", lowBound=1, upBound=n, cat=pulp.LpInteger) for i in N}
    u_g = {i: pulp.LpVariable(f"u_g_{i}", lowBound=1, upBound=n, cat=pulp.LpInteger) for i in N}
    
    # Time variables
    a = {i: pulp.LpVariable(f"a_{i}", lowBound=0) for i in N}  # Time aerial robot completes inspection at waypoint i
    g = {i: pulp.LpVariable(f"g_{i}", lowBound=0) for i in N}  # Time ground robot completes inspection at waypoint i
    
    # Objective function: Maximize the number of waypoints visited by ground robot
    model += pulp.lpSum(v_g[i] for i in N)
    
    # 1. Visit Constraints
    for i in N:
        # Outgoing edges
        model += pulp.lpSum(x_a[(i, j)] for j in list(N) + [aerial_depot_idx] if j != i) == v_a[i]
        model += pulp.lpSum(x_g[(i, j)] for j in list(N) + [ground_depot_idx] if j != i) == v_g[i]
        
        # Incoming edges
        model += pulp.lpSum(x_a[(j, i)] for j in list(N) + [aerial_depot_idx] if j != i) == v_a[i]
        model += pulp.lpSum(x_g[(j, i)] for j in list(N) + [ground_depot_idx] if j != i) == v_g[i]
    
    # 2. Depot Constraints
    model += pulp.lpSum(x_a[(aerial_depot_idx, j)] for j in N) == 1
    model += pulp.lpSum(x_a[(i, aerial_depot_idx)] for i in N) == 1
    model += pulp.lpSum(x_g[(ground_depot_idx, j)] for j in N) == 1
    model += pulp.lpSum(x_g[(i, ground_depot_idx)] for i in N) == 1
    
    # 3. Precedence Constraints
    for i in N:
        model += v_g[i] <= v_a[i]  # Ground can only visit if aerial has visited
        model += g[i] >= a[i] - M * (1 - v_g[i])  # Ground visit time after aerial visit time
    
    # 4. Subtour Elimination (MTZ formulation)
    for i in N:
        for j in N:
            if i != j:
                model += u_a[i] - u_a[j] + 1 <= (n - 1) * (1 - x_a[(i, j)])
                model += u_g[i] - u_g[j] + 1 <= (n - 1) * (1 - x_g[(i, j)])
    
    # 5. Time Constraints with Speed Considerations
    
    # Aerial robot timing between waypoints
    for i in N:
        for j in N:
            if i != j:
                model += a[j] >= a[i] + aerial_travel_times[(i, j)] - M * (1 - x_a[(i, j)])
    
    # From aerial depot to first waypoint
    for j in N:
        model += a[j] >= aerial_travel_times[(aerial_depot_idx, j)] - M * (1 - x_a[(aerial_depot_idx, j)])
    
    # Add inspection time at each waypoint
    for i in N:
        model += a[i] >= aerial_inspection_time * v_a[i]
    
    # Ground robot timing between waypoints
    for i in N:
        for j in N:
            if i != j:
                model += g[j] >= g[i] + ground_travel_times[(i, j)] - M * (1 - x_g[(i, j)])
    
    # From ground depot to first waypoint
    for j in N:
        model += g[j] >= ground_travel_times[(ground_depot_idx, j)] - M * (1 - x_g[(ground_depot_idx, j)])
    
    # Add inspection time at each waypoint
    for i in N:
        model += g[i] >= ground_inspection_time * v_g[i]
    
    # 6. Maximum Operation Time Constraints
    
    # Aerial robot maximum operation time
    for i in N:
        model += a[i] + aerial_travel_times[(i, aerial_depot_idx)] <= aerial_max_time + M * (1 - x_a[(i, aerial_depot_idx)])
    
    # Ground robot maximum operation time
    for i in N:
        model += g[i] + ground_travel_times[(i, ground_depot_idx)] <= ground_max_time + M * (1 - x_g[(i, ground_depot_idx)])
    
    # Solve the model
    model.solve(pulp.PULP_CBC_CMD(msg=True))
    
    # Check if the solution is optimal
    if model.status != pulp.LpStatusOptimal:
        return {"status": "infeasible", "message": f"Model status: {pulp.LpStatus[model.status]}"}
    
    # Extract the solution
    solution = {
        "status": "optimal",
        "objective_value": pulp.value(model.objective),
        "aerial_visited": {},
        "ground_visited": {},
        "aerial_times": {},
        "ground_times": {},
        "aerial_route": [],
        "ground_route": []
    }
    
    # Extract visited waypoints
    for i in N:
        solution["aerial_visited"][i] = 1 if pulp.value(v_a[i]) > 0.5 else 0
        solution["ground_visited"][i] = 1 if pulp.value(v_g[i]) > 0.5 else 0
        solution["aerial_times"][i] = pulp.value(a[i])
        solution["ground_times"][i] = pulp.value(g[i])
    
    # Manually extract routes from the solution
    aerial_visited = [i for i in N if solution["aerial_visited"][i] > 0.5]
    ground_visited = [i for i in N if solution["ground_visited"][i] > 0.5]
    
    # Sort waypoints by visit time
    aerial_route = [aerial_depot_idx] + sorted(aerial_visited, key=lambda i: solution["aerial_times"][i]) + [aerial_depot_idx]
    ground_route = [ground_depot_idx] + sorted(ground_visited, key=lambda i: solution["ground_times"][i]) + [ground_depot_idx]
    
    solution["aerial_route"] = aerial_route
    solution["ground_route"] = ground_route
    
    # Calculate additional solution metrics
    solution["total_aerial_time"] = max([solution["aerial_times"][i] + aerial_travel_times[(i, aerial_depot_idx)] 
                                      for i in aerial_visited]) if aerial_visited else 0
    solution["total_ground_time"] = max([solution["ground_times"][i] + ground_travel_times[(i, ground_depot_idx)] 
                                      for i in ground_visited]) if ground_visited else 0
    
    # Calculate total distance traveled
    aerial_distance = sum(distances[(aerial_route[i], aerial_route[i+1])] 
                         for i in range(len(aerial_route)-1)) if len(aerial_route) > 1 else 0
    ground_distance = sum(distances[(ground_route[i], ground_route[i+1])] 
                         for i in range(len(ground_route)-1)) if len(ground_route) > 1 else 0
    
    solution["aerial_distance"] = aerial_distance
    solution["ground_distance"] = ground_distance
    
    print("Solution Summary:")
    print(f"Objective value: {solution['objective_value']}")
    print(f"Aerial robot visited waypoints: {sum(solution['aerial_visited'].values())}")
    print(f"Ground robot visited waypoints: {sum(solution['ground_visited'].values())}")
    print(f"Aerial robot route: {aerial_route}")
    print(f"Ground robot route: {ground_route}")
    print(f"Total aerial robot time: {solution['total_aerial_time']:.2f} minutes")
    print(f"Total ground robot time: {solution['total_ground_time']:.2f} minutes")
    print(f"Total aerial robot distance: {solution['aerial_distance']:.2f} units")
    print(f"Total ground robot distance: {solution['ground_distance']:.2f} units")
    
    return solution

def visualize_solution(solution, waypoints, aerial_depot, ground_depot, 
                      aerial_speed, ground_speed, save_path=None):
    """
    Visualize the solution with speed information.
    
    Parameters:
    -----------
    solution : dict
        The solution returned by solve_robot_inspection_problem
    waypoints : list of tuples (x, y)
        Coordinates of inspection waypoints
    aerial_depot : tuple (x, y)
        Coordinates of aerial robot depot
    ground_depot : tuple (x, y)
        Coordinates of ground robot depot
    aerial_speed : float
        Speed of aerial robot (distance units per minute)
    ground_speed : float
        Speed of ground robot (distance units per minute)
    save_path : str, optional
        Path to save the visualization
    """
    if solution["status"] != "optimal":
        print("No optimal solution to visualize.")
        return
    
    # Set matplotlib config directory to a temporary directory to avoid permission issues
    temp_dir = tempfile.mkdtemp()
    os.environ['MPLCONFIGDIR'] = temp_dir
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Number of waypoints
    n = len(waypoints)
    
    # Plot waypoints
    waypoint_x = [waypoints[i][0] for i in range(n)]
    waypoint_y = [waypoints[i][1] for i in range(n)]
    
    # Determine which waypoints are visited by which robots
    waypoint_status = []
    for i in range(n):
        if solution["ground_visited"].get(i, 0) > 0.5:
            waypoint_status.append('both')
        elif solution["aerial_visited"].get(i, 0) > 0.5:
            waypoint_status.append('aerial')
        else:
            waypoint_status.append('none')
    
    # Plot waypoints with different colors based on status
    for i in range(n):
        if waypoint_status[i] == 'both':
            ax.scatter(waypoint_x[i], waypoint_y[i], s=100, color='green', 
                       edgecolor='black', zorder=3, label='_nolegend_')
            ax.text(waypoint_x[i], waypoint_y[i]+0.2, f"{i}", 
                    ha='center', va='center', fontsize=10)
        elif waypoint_status[i] == 'aerial':
            ax.scatter(waypoint_x[i], waypoint_y[i], s=100, color='lightblue', 
                       edgecolor='black', zorder=3, label='_nolegend_')
            ax.text(waypoint_x[i], waypoint_y[i]+0.2, f"{i}", 
                    ha='center', va='center', fontsize=10)
        else:
            ax.scatter(waypoint_x[i], waypoint_y[i], s=100, color='lightgray', 
                       edgecolor='black', zorder=3, label='_nolegend_')
            ax.text(waypoint_x[i], waypoint_y[i]+0.2, f"{i}", 
                    ha='center', va='center', fontsize=10)
    
    # Plot depots
    ax.scatter(aerial_depot[0], aerial_depot[1], s=200, color='blue', 
               marker='^', edgecolor='black', zorder=4, label='Aerial Depot')
    ax.scatter(ground_depot[0], ground_depot[1], s=200, color='red', 
               marker='s', edgecolor='black', zorder=4, label='Ground Depot')
    
    # Plot aerial robot route
    aerial_route_coords = []
    aerial_depot_idx = -1
    for i in solution["aerial_route"]:
        if i == aerial_depot_idx:
            aerial_route_coords.append(aerial_depot)
        else:
            aerial_route_coords.append(waypoints[i])
            
    aerial_x = [coord[0] for coord in aerial_route_coords]
    aerial_y = [coord[1] for coord in aerial_route_coords]
    ax.plot(aerial_x, aerial_y, 'b-', linewidth=2, alpha=0.7, zorder=2)
    
    # Add arrows and time labels for aerial route
    for i in range(len(aerial_x) - 1):
        # Add arrow
        ax.annotate('', xy=(aerial_x[i+1], aerial_y[i+1]), 
                    xytext=(aerial_x[i], aerial_y[i]),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.7))
        
        # Add time label for waypoints (not depots)
        idx = solution["aerial_route"][i]
        if idx != aerial_depot_idx and i > 0:
            time_val = solution["aerial_times"][idx]
            midpoint_x = (aerial_x[i] + aerial_x[i-1]) / 2
            midpoint_y = (aerial_y[i] + aerial_y[i-1]) / 2
            ax.text(midpoint_x, midpoint_y, f"{time_val:.1f}m", 
                    color='blue', fontsize=8, ha='center', va='bottom')
    
    # Plot ground robot route
    ground_route_coords = []
    ground_depot_idx = -2
    for i in solution["ground_route"]:
        if i == ground_depot_idx:
            ground_route_coords.append(ground_depot)
        else:
            ground_route_coords.append(waypoints[i])
            
    ground_x = [coord[0] for coord in ground_route_coords]
    ground_y = [coord[1] for coord in ground_route_coords]
    ax.plot(ground_x, ground_y, 'r-', linewidth=2, alpha=0.7, zorder=2)
    
    # Add arrows and time labels for ground route
    for i in range(len(ground_x) - 1):
        # Add arrow
        ax.annotate('', xy=(ground_x[i+1], ground_y[i+1]), 
                    xytext=(ground_x[i], ground_y[i]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))
        
        # Add time label for waypoints (not depots)
        idx = solution["ground_route"][i]
        if idx != ground_depot_idx and i > 0:
            time_val = solution["ground_times"][idx]
            midpoint_x = (ground_x[i] + ground_x[i-1]) / 2
            midpoint_y = (ground_y[i] + ground_y[i-1]) / 2
            ax.text(midpoint_x, midpoint_y, f"{time_val:.1f}m", 
                    color='red', fontsize=8, ha='center', va='top')
    
    # Create legend with custom labels
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, 
                  label=f'Aerial Robot Path (Speed: {aerial_speed} units/min)'),
        plt.Line2D([0], [0], color='red', lw=2, 
                  label=f'Ground Robot Path (Speed: {ground_speed} units/min)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                   markersize=10, label='Aerial Depot'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                   markersize=10, label='Ground Depot'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='Visited by Both'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='Visited by Aerial Only'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                   markersize=10, label='Not Visited')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    # Add solution summary text
    summary_text = (
        f"Objective: {solution['objective_value']} waypoints inspected\n"
        f"Aerial time: {solution['total_aerial_time']:.1f} minutes\n"
        f"Ground time: {solution['total_ground_time']:.1f} minutes\n"
        f"Aerial distance: {solution['aerial_distance']:.1f} units\n"
        f"Ground distance: {solution['ground_distance']:.1f} units"
    )
    ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Set plot title and labels
    ax.set_title('Multi-Robot Inspection Plan', fontsize=16)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits with some padding
    all_x = waypoint_x + [aerial_depot[0], ground_depot[0]]
    all_y = waypoint_y + [aerial_depot[1], ground_depot[1]]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    padding = max(x_max - x_min, y_max - y_min) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def run_example():
    """Run an example to demonstrate the robot inspection problem with speed considerations."""
    # Create waypoints
    waypoints = [(1, 5), (3, 3), (5, 5), (7, 3), (9, 5), (5, 7)]
    
    # Define depot locations
    aerial_depot = (0, 0)
    ground_depot = (10, 0)
    
    # Define robot speeds (distance units per minute)
    aerial_speed = 5.0  # Faster aerial robot
    ground_speed = 2.0  # Slower ground robot
    
    # Define operation time limits (minutes)
    aerial_max_time = 30.0
    ground_max_time = 60.0
    
    # Define inspection times (minutes)
    aerial_inspection_time = 1.0
    ground_inspection_time = 3.0
    
    print("Robot Inspection Problem with Speed Considerations")
    print(f"Number of waypoints: {len(waypoints)}")
    print(f"Aerial robot speed: {aerial_speed} units/minute")
    print(f"Ground robot speed: {ground_speed} units/minute")
    print(f"Aerial robot max operation time: {aerial_max_time} minutes")
    print(f"Ground robot max operation time: {ground_max_time} minutes")
    print(f"Aerial inspection time: {aerial_inspection_time} minutes/waypoint")
    print(f"Ground inspection time: {ground_inspection_time} minutes/waypoint")
    
    # Solve the problem
    solution = solve_robot_inspection_problem(
        waypoints, aerial_depot, ground_depot,
        aerial_speed, ground_speed,
        aerial_max_time, ground_max_time,
        aerial_inspection_time, ground_inspection_time
    )
    
    # Visualize the solution
    if solution["status"] == "optimal":
        visualize_solution(
            solution, waypoints, aerial_depot, ground_depot,
            aerial_speed, ground_speed, "inspection_plan.png"
        )
    
if __name__ == "__main__":
    run_example()