import pulp
import numpy as np


def solve_multi_robot_inspection_problem(
        waypoints, 
        aerial_depot, 
        ground_depot,
        aerial_speed,        # Speed for all aerial robots
        ground_speed,        # Speed for all ground robots
        aerial_max_time,     # Max operation time for all aerial robots
        ground_max_time,     # Max operation time for all ground robots
        aerial_inspection_time, 
        ground_inspection_time,
        num_aerial_robots,   # Number of aerial robots
        num_ground_robots):  # Number of ground robots
    """
    Simplified solver for the multi-robot inspection problem.
    """
    # Number of waypoints and robots
    n = len(waypoints)
    k_max = num_aerial_robots
    l_max = num_ground_robots
    
    print(f"Problem setup: {n} waypoints, {k_max} aerial robots, {l_max} ground robots")
    
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
    aerial_travel_times = {k_v: v / aerial_speed for k_v, v in distances.items()}
    ground_travel_times = {k_v: v / ground_speed for k_v, v in distances.items()}
    
    # Create the optimization model
    model = pulp.LpProblem("MultiRobotInspection", pulp.LpMaximize)
    
    # Calculate appropriate Big-M values
    max_aerial_travel = max(aerial_travel_times.values())
    M_A = aerial_max_time + max_aerial_travel + aerial_inspection_time
    
    max_ground_travel = max(ground_travel_times.values())
    M_G = ground_max_time + max_ground_travel + ground_inspection_time
    
    
    # Sets for indexing
    N = range(n)  # Waypoints
    K = range(k_max)  # Aerial robots
    L = range(l_max)  # Ground robots
    
    # SIMPLIFIED APPROACH: Use assignment variables instead of routing variables
    # This greatly reduces complexity while still ensuring feasibility
    
    # Decision Variables
    # Binary variable: 1 if aerial robot k visits waypoint i
    v_a = {(i, k): pulp.LpVariable(f"v_a_{i}_{k}", cat=pulp.LpBinary) for i in N for k in K}
    
    # Binary variable: 1 if ground robot l visits waypoint i
    v_g = {(i, l): pulp.LpVariable(f"v_g_{i}_{l}", cat=pulp.LpBinary) for i in N for l in L}
    
    # Time when aerial robot k completes inspection at waypoint i
    a_time = {(i, k): pulp.LpVariable(f"a_time_{i}_{k}", lowBound=0) for i in N for k in K}
    
    # Time when ground robot l completes inspection at waypoint i
    g_time = {(i, l): pulp.LpVariable(f"g_time_{i}_{l}", lowBound=0) for i in N for l in L}
    
    # Binary variables to indicate if a robot is used
    use_aerial = {k: pulp.LpVariable(f"use_aerial_{k}", cat=pulp.LpBinary) for k in K}
    use_ground = {l: pulp.LpVariable(f"use_ground_{l}", cat=pulp.LpBinary) for l in L}
    
    # Used for the relationship between aerial and ground robots
    inspect_after = {(i, k, l): pulp.LpVariable(f"inspect_after_{i}_{k}_{l}", cat=pulp.LpBinary) 
                     for i in N for k in K for l in L}
    
    # Objective: Maximize the number of waypoints visited by ground robots
    model += pulp.lpSum(v_g[(i, l)] for i in N for l in L)
    
    # Constraints:
    
    # 1. Each waypoint can be visited by at most one aerial robot
    for i in N:
        model += pulp.lpSum(v_a[(i, k)] for k in K) <= 1
    
    # 2. Each waypoint can be visited by at most one ground robot
    for i in N:
        model += pulp.lpSum(v_g[(i, l)] for l in L) <= 1
    
    
    # 4. Each aerial robot can visit at most n waypoints (simplification)
    for k in K:
        model += pulp.lpSum(v_a[(i, k)] for i in N) <= n * use_aerial[k]
        model += pulp.lpSum(v_a[(i, k)] for i in N) >= use_aerial[k]  # Robot is used if it visits at least one waypoint
    
    # 5. Each ground robot can visit at most n waypoints (simplification)
    for l in L:
        model += pulp.lpSum(v_g[(i, l)] for i in N) <= n * use_ground[l]
        model += pulp.lpSum(v_g[(i, l)] for i in N) >= use_ground[l]  # Robot is used if it visits at least one waypoint
    
    # 6. Time constraints for aerial robots
    for k in K:
        for i in N:
            # If robot k visits waypoint i, it must spend at least inspection time there
            model += a_time[(i, k)] >= aerial_inspection_time * v_a[(i, k)]
            
            # Total time must be within max operation time
            # We estimate travel time based on distance from depot
            model += a_time[(i, k)] + aerial_travel_times[(i, aerial_depot_idx)] * v_a[(i, k)] <= aerial_max_time + M_A * (1 - v_a[(i, k)])
    
    # 7. Time constraints for ground robots
    for l in L:
        for i in N:
            # If robot l visits waypoint i, it must spend at least inspection time there
            model += g_time[(i, l)] >= ground_inspection_time * v_g[(i, l)]
            
            # Total time must be within max operation time
            # We estimate travel time based on distance from depot
            model += g_time[(i, l)] + ground_travel_times[(i, ground_depot_idx)] * v_g[(i, l)] <= ground_max_time + M_G * (1 - v_g[(i, l)])
    
    # A ground robot can only visit a waypoint if an aerial robot has visited it
    for i in N:
        for l in L:
            model += v_g[(i, l)] <= pulp.lpSum(v_a[(i, k)] for k in K)


    # 8. Precedence constraints: ground robot can only inspect after aerial robot has inspected
    for i in N:
        for k in K:
            for l in L:
                # inspect_after[i,k,l] = 1 if aerial robot k inspects i and then ground robot l inspects i
                model += inspect_after[(i, k, l)] <= v_a[(i, k)]
                model += inspect_after[(i, k, l)] <= v_g[(i, l)]
                model += inspect_after[(i, k, l)] >= v_a[(i, k)] + v_g[(i, l)] - 1
                
                # If ground robot l visits after aerial robot k, ensure proper timing
                model += g_time[(i, l)] >= a_time[(i, k)] - M_G * (1 - inspect_after[(i, k, l)])
    
    # 9. Simple approximate route length constraints for aerial robots
    for k in K:
        # Simple constraint: total estimated travel time plus inspection time
        total_travel_time = pulp.lpSum(
            v_a[(i, k)] * (
                aerial_travel_times[(aerial_depot_idx, i)] +  # From depot to waypoint
                aerial_travel_times[(i, aerial_depot_idx)]    # From waypoint back to depot
            ) / 2  # Average trip (this is a simplification) 
            for i in N
        )
        total_inspection_time = pulp.lpSum(v_a[(i, k)] * aerial_inspection_time for i in N)
        
        model += total_travel_time + total_inspection_time <= aerial_max_time + M_A * (1 - use_aerial[k])
    
    # 10. Simple approximate route length constraints for ground robots
    for l in L:
        # Simple constraint: total estimated travel time plus inspection time
        total_travel_time = pulp.lpSum(
            v_g[(i, l)] * (
                ground_travel_times[(ground_depot_idx, i)] +  # From depot to waypoint
                ground_travel_times[(i, ground_depot_idx)]    # From waypoint back to depot
            ) / 2  # Average trip (this is a simplification)
            for i in N
        )
        total_inspection_time = pulp.lpSum(v_g[(i, l)] * ground_inspection_time for i in N)
        
        model += total_travel_time + total_inspection_time <= ground_max_time + M_G * (1 - use_ground[l])
    
    # Solve the model with debugging on
    print("Solving the simplified optimization model...")
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=500)
    model.solve(solver)
    
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
        "aerial_routes": {},
        "ground_routes": {}
    }
    
    # Extract visited waypoints for each robot
    for k in K:
        solution["aerial_visited"][k] = {}
        solution["aerial_times"][k] = {}
        
        # Determine which waypoints are visited by aerial robot k
        for i in N:
            if pulp.value(v_a[(i, k)]) > 0.5:
                solution["aerial_visited"][k][i] = 1
                solution["aerial_times"][k][i] = pulp.value(a_time[(i, k)])
            else:
                solution["aerial_visited"][k][i] = 0
    
    for l in L:
        solution["ground_visited"][l] = {}
        solution["ground_times"][l] = {}
        
        # Determine which waypoints are visited by ground robot l
        for i in N:
            if pulp.value(v_g[(i, l)]) > 0.5:
                solution["ground_visited"][l][i] = 1
                solution["ground_times"][l][i] = pulp.value(g_time[(i, l)])
            else:
                solution["ground_visited"][l][i] = 0
    
    # Generate approximate routes for visualization
    # This is a simplification: in reality, we'd need to solve a TSP for each robot's assigned waypoints
    # But for visualization purposes, we'll just create routes based on nearest neighbor
    
    for k in K:
        if pulp.value(use_aerial[k]) < 0.5:
            solution["aerial_routes"][k] = [aerial_depot_idx, aerial_depot_idx]  # Empty route
            continue
            
        visited_waypoints = [i for i in N if solution["aerial_visited"][k].get(i, 0) > 0.5]
        
        if not visited_waypoints:
            solution["aerial_routes"][k] = [aerial_depot_idx, aerial_depot_idx]  # Empty route
            continue
            
        # Sort waypoints by completion time to get an approximate route
        sorted_waypoints = sorted(visited_waypoints, key=lambda i: solution["aerial_times"][k][i])
        
        # Create route: depot -> sorted waypoints -> depot
        route = [aerial_depot_idx] + sorted_waypoints + [aerial_depot_idx]
        solution["aerial_routes"][k] = route
    
    for l in L:
        if pulp.value(use_ground[l]) < 0.5:
            solution["ground_routes"][l] = [ground_depot_idx, ground_depot_idx]  # Empty route
            continue
            
        visited_waypoints = [i for i in N if solution["ground_visited"][l].get(i, 0) > 0.5]
        
        if not visited_waypoints:
            solution["ground_routes"][l] = [ground_depot_idx, ground_depot_idx]  # Empty route
            continue
            
        # Sort waypoints by completion time to get an approximate route
        sorted_waypoints = sorted(visited_waypoints, key=lambda i: solution["ground_times"][l][i])
        
        # Create route: depot -> sorted waypoints -> depot
        route = [ground_depot_idx] + sorted_waypoints + [ground_depot_idx]
        solution["ground_routes"][l] = route
    
    # Calculate route distances
    solution["aerial_distances"] = {}
    solution["ground_distances"] = {}
    solution["total_aerial_times"] = {}
    solution["total_ground_times"] = {}
    
    for k in K:
        aerial_route = solution["aerial_routes"][k]
        if len(aerial_route) <= 2:  # Only depot-depot
            solution["aerial_distances"][k] = 0
            solution["total_aerial_times"][k] = 0
            continue
            
        # Calculate distance
        aerial_distance = sum(distances[(aerial_route[i], aerial_route[i+1])] 
                             for i in range(len(aerial_route)-1))
        solution["aerial_distances"][k] = aerial_distance
        
        # Calculate time (approximate)
        aerial_time = aerial_distance / aerial_speed + len(aerial_route) * aerial_inspection_time
        solution["total_aerial_times"][k] = aerial_time
    
    for l in L:
        ground_route = solution["ground_routes"][l]
        if len(ground_route) <= 2:  # Only depot-depot
            solution["ground_distances"][l] = 0
            solution["total_ground_times"][l] = 0
            continue
            
        # Calculate distance
        ground_distance = sum(distances[(ground_route[i], ground_route[i+1])] 
                             for i in range(len(ground_route)-1))
        solution["ground_distances"][l] = ground_distance
        
        # Calculate time (approximate)
        ground_time = ground_distance / ground_speed + len(ground_route) * ground_inspection_time
        solution["total_ground_times"][l] = ground_time
    
    # Print summary
    print("\nSolution Summary:")
    print(f"Objective value: {solution['objective_value']} waypoints inspected")
    
    for k in K:
        aerial_visited_count = sum(solution['aerial_visited'][k].values())
        print(f"\nAerial robot {k} visited {aerial_visited_count} waypoints")
        print(f"  Route: {solution['aerial_routes'][k]}")
        print(f"  Total distance: {solution['aerial_distances'][k]:.2f} units")
        print(f"  Approx. total time: {solution['total_aerial_times'][k]:.2f} minutes")
    
    for l in L:
        ground_visited_count = sum(solution['ground_visited'][l].values())
        print(f"\nGround robot {l} visited {ground_visited_count} waypoints")
        print(f"  Route: {solution['ground_routes'][l]}")
        print(f"  Total distance: {solution['ground_distances'][l]:.2f} units")
        print(f"  Approx. total time: {solution['total_ground_times'][l]:.2f} minutes")
    
    return solution

def run_example():

    from solver_code.figures import visualize_multi_robot_solution

    """Run an example using the simplified multi-robot inspection solver."""
    # Create waypoints
    waypoints = [(1, 5), (3, 3), (5, 5), (7, 3), (9, 5), (5, 7), (2, 8), (8, 8), (3, 1), (7, 1)]
    
    # Define depot locations
    aerial_depot = (0, 0)
    ground_depot = (10, 0)
    
    # Define robot parameters (same for all robots of each type)
    aerial_speed = 5.0
    ground_speed = 2.0
    aerial_max_time = 30.0
    ground_max_time = 60.0
    
    # Number of robots
    num_aerial_robots = 10
    num_ground_robots = 3
    
    # Define inspection times (minutes)
    aerial_inspection_time = 1.0
    ground_inspection_time = 3.0
    
    print("Multi-Robot Inspection Problem")
    print(f"Number of waypoints: {len(waypoints)}")
    print(f"Number of aerial robots: {num_aerial_robots}")
    print(f"Number of ground robots: {num_ground_robots}")
    print(f"Aerial robot speed: {aerial_speed} units/minute")
    print(f"Ground robot speed: {ground_speed} units/minute")
    print(f"Aerial robot max operation time: {aerial_max_time} minutes")
    print(f"Ground robot max operation time: {ground_max_time} minutes")
    
    # Solve the problem
    solution = solve_multi_robot_inspection_problem(
        waypoints, aerial_depot, ground_depot,
        aerial_speed, ground_speed,
        aerial_max_time, ground_max_time,
        aerial_inspection_time, ground_inspection_time,
        num_aerial_robots, num_ground_robots
    )
    
    # Visualize the solution
    if solution["status"] == "optimal":
        visualize_multi_robot_solution(
            solution, waypoints, aerial_depot, ground_depot,
            aerial_speed, ground_speed, "multi_robot_inspection_plan.png"
        )

if __name__ == "__main__":
    run_example()