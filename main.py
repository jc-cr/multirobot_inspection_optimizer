import pulp
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import os
from matplotlib.lines import Line2D

class MultiRobotInspectionPlanner:
    def __init__(self, map_size=(100, 100), num_waypoints=10, seed=None):
        """
        Initialize the planner with map and waypoint parameters.
        
        Args:
            map_size: Tuple of (width, height) for the 2D map
            num_waypoints: Number of waypoints to generate
            seed: Random seed for reproducibility
        """
        self.map_size = map_size
        self.num_waypoints = num_waypoints
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Generate waypoints (random locations by default)
        self.waypoints = self._generate_waypoints()
        
        # Define depot locations (corners of the map by default)
        self.depot_aerial = (0, 0)  # Bottom-left corner
        self.depot_ground = (map_size[0], map_size[1])  # Top-right corner
        
        # Robot parameters
        self.t_aerial = 2.0  # Time for aerial inspection at each waypoint
        self.t_ground = 5.0  # Time for ground inspection at each waypoint
        self.b_aerial = 0.1  # Battery depletion rate for aerial robot
        self.b_ground = 0.05  # Battery depletion rate for ground robot
        self.B_aerial = 100.0  # Initial battery level for aerial robot
        self.B_ground = 100.0  # Initial battery level for ground robot
        
        # Solver results
        self.solution = None
        self.aerial_route = None
        self.ground_route = None
        self.aerial_times = None
        self.ground_times = None
        
    def _generate_waypoints(self):
        """Generate random waypoints within the map."""
        waypoints = []
        for i in range(self.num_waypoints):
            x = np.random.uniform(0, self.map_size[0])
            y = np.random.uniform(0, self.map_size[1])
            waypoints.append((x, y))
        return waypoints
    
    def set_waypoints(self, waypoints):
        """Set custom waypoints."""
        self.waypoints = waypoints
        self.num_waypoints = len(waypoints)
    
    def set_depots(self, aerial_depot, ground_depot):
        """Set custom depot locations."""
        self.depot_aerial = aerial_depot
        self.depot_ground = ground_depot
    
    def set_robot_params(self, t_aerial, t_ground, b_aerial, b_ground, B_aerial, B_ground):
        """Set custom robot parameters."""
        self.t_aerial = t_aerial
        self.t_ground = t_ground
        self.b_aerial = b_aerial
        self.b_ground = b_ground
        self.B_aerial = B_aerial
        self.B_ground = B_ground
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def solve(self):
        """Solve the MIP problem to find optimal inspection paths."""
        # Create problem
        prob = pulp.LpProblem("MultiRobotInspection", pulp.LpMaximize)
        
        # Create sets
        N = list(range(self.num_waypoints))
        all_nodes_aerial = [self.num_waypoints] + list(N)  # Include depot
        all_nodes_ground = [self.num_waypoints + 1] + list(N)  # Include depot
        
        # Calculate travel costs (distances)
        all_points = self.waypoints + [self.depot_aerial, self.depot_ground]
        c_aerial = {}
        c_ground = {}
        
        for i in all_nodes_aerial:
            for j in all_nodes_aerial:
                if i != j:
                    i_point = all_points[i] if i < self.num_waypoints else all_points[self.num_waypoints]
                    j_point = all_points[j] if j < self.num_waypoints else all_points[self.num_waypoints]
                    c_aerial[(i, j)] = self._calculate_distance(i_point, j_point)
        
        for i in all_nodes_ground:
            for j in all_nodes_ground:
                if i != j:
                    i_point = all_points[i] if i < self.num_waypoints else all_points[self.num_waypoints + 1]
                    j_point = all_points[j] if j < self.num_waypoints else all_points[self.num_waypoints + 1]
                    c_ground[(i, j)] = self._calculate_distance(i_point, j_point)
        
        # Create decision variables
        depot_aerial_idx = self.num_waypoints
        depot_ground_idx = self.num_waypoints + 1
        
        # x_ij variables: 1 if robot travels from i to j, 0 otherwise
        x_aerial = {}
        for i in all_nodes_aerial:
            for j in all_nodes_aerial:
                if i != j:
                    x_aerial[(i, j)] = pulp.LpVariable(f"x_aerial_{i}_{j}", cat='Binary')
        
        x_ground = {}
        for i in all_nodes_ground:
            for j in all_nodes_ground:
                if i != j:
                    x_ground[(i, j)] = pulp.LpVariable(f"x_ground_{i}_{j}", cat='Binary')
        
        # v_i variables: 1 if waypoint i is visited, 0 otherwise
        v_aerial = {}
        v_ground = {}
        for i in N:
            v_aerial[i] = pulp.LpVariable(f"v_aerial_{i}", cat='Binary')
            v_ground[i] = pulp.LpVariable(f"v_ground_{i}", cat='Binary')
        
        # u_i variables: position in the route (for subtour elimination)
        u_aerial = {}
        u_ground = {}
        for i in N:
            u_aerial[i] = pulp.LpVariable(f"u_aerial_{i}", lowBound=1, upBound=self.num_waypoints, cat='Integer')
            u_ground[i] = pulp.LpVariable(f"u_ground_{i}", lowBound=1, upBound=self.num_waypoints, cat='Integer')
        
        # T_i variables: time when robot completes inspection at waypoint i
        T_aerial = {}
        T_ground = {}
        big_M = 1000  # A large constant
        
        for i in N:
            T_aerial[i] = pulp.LpVariable(f"T_aerial_{i}", lowBound=0, cat='Continuous')
            T_ground[i] = pulp.LpVariable(f"T_ground_{i}", lowBound=0, cat='Continuous')
        
        # Objective: Maximize number of waypoints visited by ground robot
        prob += pulp.lpSum(v_ground[i] for i in N)
        
        # Constraints
        
        # 1. Visit Indicators
        for i in N:
            prob += pulp.lpSum(x_aerial[(i, j)] for j in all_nodes_aerial if j != i) == v_aerial[i]
            prob += pulp.lpSum(x_ground[(i, j)] for j in all_nodes_ground if j != i) == v_ground[i]
        
        # 2. Flow Conservation
        for i in N:
            prob += pulp.lpSum(x_aerial[(i, j)] for j in all_nodes_aerial if j != i) == pulp.lpSum(x_aerial[(j, i)] for j in all_nodes_aerial if j != i)
            prob += pulp.lpSum(x_ground[(i, j)] for j in all_nodes_ground if j != i) == pulp.lpSum(x_ground[(j, i)] for j in all_nodes_ground if j != i)
        
        # 3. Precedence Constraint
        for i in N:
            prob += v_ground[i] <= v_aerial[i]
            prob += T_ground[i] >= T_aerial[i] + self.t_aerial - big_M * (1 - v_ground[i])
        
        # 4. Subtour Elimination
        for i in N:
            for j in N:
                if i != j:
                    prob += u_aerial[i] - u_aerial[j] + self.num_waypoints * x_aerial[(i, j)] <= self.num_waypoints - 1 + self.num_waypoints * (1 - v_aerial[i]) + self.num_waypoints * (1 - v_aerial[j])
                    prob += u_ground[i] - u_ground[j] + self.num_waypoints * x_ground[(i, j)] <= self.num_waypoints - 1 + self.num_waypoints * (1 - v_ground[i]) + self.num_waypoints * (1 - v_ground[j])
        
        # 5. Battery Constraints
        prob += pulp.lpSum(c_aerial[(i, j)] * x_aerial[(i, j)] for i, j in product(all_nodes_aerial, all_nodes_aerial) if i != j) + pulp.lpSum(self.t_aerial * v_aerial[i] for i in N) <= self.B_aerial / self.b_aerial
        prob += pulp.lpSum(c_ground[(i, j)] * x_ground[(i, j)] for i, j in product(all_nodes_ground, all_nodes_ground) if i != j) + pulp.lpSum(self.t_ground * v_ground[i] for i in N) <= self.B_ground / self.b_ground
        
        # 6. Depot Constraints
        prob += pulp.lpSum(x_aerial[(depot_aerial_idx, j)] for j in N) == 1
        prob += pulp.lpSum(x_aerial[(i, depot_aerial_idx)] for i in N) == 1
        prob += pulp.lpSum(x_ground[(depot_ground_idx, j)] for j in N) == 1
        prob += pulp.lpSum(x_ground[(i, depot_ground_idx)] for i in N) == 1
        
        # 7. Time Tracking
        for i in N:
            for j in N:
                if i != j:
                    prob += T_aerial[j] >= T_aerial[i] + self.t_aerial + c_aerial[(i, j)] - big_M * (1 - x_aerial[(i, j)])
                    prob += T_ground[j] >= T_ground[i] + self.t_ground + c_ground[(i, j)] - big_M * (1 - x_ground[(i, j)])
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=True))
        
        # Check if a solution was found
        if pulp.LpStatus[prob.status] == 'Optimal':
            self.solution = prob
            self._extract_solution()
            return True
        else:
            print(f"No solution found. Status: {pulp.LpStatus[prob.status]}")
            return False
    
    def _extract_solution(self):
        """Extract the solution from the solved model."""
        if not self.solution:
            return None
        
        N = list(range(self.num_waypoints))
        depot_aerial_idx = self.num_waypoints
        depot_ground_idx = self.num_waypoints + 1
        
        # Extract visited waypoints
        visited_aerial = [i for i in N if pulp.value(self.solution.variablesDict()[f"v_aerial_{i}"]) > 0.5]
        visited_ground = [i for i in N if pulp.value(self.solution.variablesDict()[f"v_ground_{i}"]) > 0.5]
        
        # Extract routes
        aerial_route = []
        ground_route = []
        
        # Extract aerial route
        current = depot_aerial_idx
        while True:
            next_waypoint = None
            for i in N + [depot_aerial_idx]:
                if i != current and i in visited_aerial + [depot_aerial_idx]:
                    if pulp.value(self.solution.variablesDict()[f"x_aerial_{current}_{i}"]) > 0.5:
                        next_waypoint = i
                        break
            
            if next_waypoint is None or next_waypoint == depot_aerial_idx and len(aerial_route) > 0:
                aerial_route.append(depot_aerial_idx)
                break
            
            aerial_route.append(next_waypoint)
            current = next_waypoint
        
        # Extract ground route
        current = depot_ground_idx
        while True:
            next_waypoint = None
            for i in N + [depot_ground_idx]:
                if i != current and i in visited_ground + [depot_ground_idx]:
                    if pulp.value(self.solution.variablesDict()[f"x_ground_{current}_{i}"]) > 0.5:
                        next_waypoint = i
                        break
            
            if next_waypoint is None or next_waypoint == depot_ground_idx and len(ground_route) > 0:
                ground_route.append(depot_ground_idx)
                break
            
            ground_route.append(next_waypoint)
            current = next_waypoint
        
        # Extract completion times
        aerial_times = {i: pulp.value(self.solution.variablesDict()[f"T_aerial_{i}"]) for i in visited_aerial}
        ground_times = {i: pulp.value(self.solution.variablesDict()[f"T_ground_{i}"]) for i in visited_ground}
        
        self.aerial_route = aerial_route
        self.ground_route = ground_route
        self.aerial_times = aerial_times
        self.ground_times = ground_times
        
        return {
            'aerial_route': aerial_route,
            'ground_route': ground_route,
            'aerial_times': aerial_times,
            'ground_times': ground_times,
            'visited_aerial': visited_aerial,
            'visited_ground': visited_ground
        }
    
    def get_results(self):
        """Get the results of the optimization."""
        if not self.solution:
            return "No solution available. Run solve() first."
        
        results = self._extract_solution()
        
        visited_aerial = results['visited_aerial']
        visited_ground = results['visited_ground']
        
        summary = {
            'total_waypoints': self.num_waypoints,
            'visited_by_aerial': len(visited_aerial),
            'visited_by_ground': len(visited_ground),
            'aerial_route': self.aerial_route,
            'ground_route': self.ground_route,
            'objective_value': pulp.value(self.solution.objective)
        }
        
        return summary

class PathVisualizer:
    def __init__(self, planner):
        """
        Initialize the visualizer with the inspection planner.
        
        Args:
            planner: MultiRobotInspectionPlanner instance with solved paths
        """
        self.planner = planner
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.map_size = planner.map_size
        
        # Colors
        self.aerial_color = 'blue'
        self.ground_color = 'green'
        self.waypoint_color = 'gray'
        self.depot_color = 'red'
        self.visited_aerial_color = 'lightblue'
        self.visited_ground_color = 'lightgreen'
        self.visited_both_color = 'purple'
        
        # Animation objects
        self.animation = None
        
    def prepare_visualization(self):
        """Prepare the map visualization with waypoints and depots."""
        # Set up the plot
        self.ax.clear()
        self.ax.set_xlim([-10, self.map_size[0] + 10])
        self.ax.set_ylim([-10, self.map_size[1] + 10])
        self.ax.set_aspect('equal')
        self.ax.set_title('Multi-Robot Inspection Path Planning')
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        
        # Plot waypoints
        for i, waypoint in enumerate(self.planner.waypoints):
            self.ax.plot(waypoint[0], waypoint[1], 'o', color=self.waypoint_color, markersize=8)
            self.ax.text(waypoint[0] + 2, waypoint[1] + 2, str(i), fontsize=9)
        
        # Plot depots
        self.ax.plot(self.planner.depot_aerial[0], self.planner.depot_aerial[1], 's', 
                     color=self.depot_color, markersize=10)
        self.ax.text(self.planner.depot_aerial[0] + 2, self.planner.depot_aerial[1] + 2, 
                     "Aerial Depot", fontsize=9)
        
        self.ax.plot(self.planner.depot_ground[0], self.planner.depot_ground[1], 's', 
                     color=self.depot_color, markersize=10)
        self.ax.text(self.planner.depot_ground[0] + 2, self.planner.depot_ground[1] + 2, 
                     "Ground Depot", fontsize=9)
        
        # Create legend elements
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.waypoint_color, 
                   markersize=8, label='Waypoint'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=self.depot_color, 
                   markersize=8, label='Depot'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.aerial_color, 
                   markersize=8, label='Aerial Robot'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.ground_color, 
                   markersize=8, label='Ground Robot'),
            Line2D([0], [0], color=self.aerial_color, lw=2, label='Aerial Path'),
            Line2D([0], [0], color=self.ground_color, lw=2, label='Ground Path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.visited_aerial_color, 
                   markersize=8, label='Visited by Aerial'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.visited_ground_color, 
                   markersize=8, label='Visited by Ground'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.visited_both_color, 
                   markersize=8, label='Visited by Both')
        ]
        
        # Add the legend
        self.ax.legend(handles=legend_elements, loc='upper right')
        
    def _get_point_coordinates(self, point_idx):
        """Get coordinates for a point (waypoint or depot)."""
        if point_idx < self.planner.num_waypoints:
            return self.planner.waypoints[point_idx]
        elif point_idx == self.planner.num_waypoints:
            return self.planner.depot_aerial
        else:
            return self.planner.depot_ground
        
    def _interpolate_path(self, route, num_steps=20):
        """Interpolate a path to create smooth animation."""
        points = [self._get_point_coordinates(point_idx) for point_idx in route]
        interpolated_path = []
        
        for i in range(len(points) - 1):
            start_point = np.array(points[i])
            end_point = np.array(points[i+1])
            
            for step in range(num_steps):
                t = step / num_steps
                point = start_point * (1 - t) + end_point * t
                interpolated_path.append(point)
                
        return interpolated_path
    
    def _calculate_times(self, route, times, robot_type):
        """Calculate timestamps for each position in the interpolated path."""
        if not route:  # If no route is provided
            return []
            
        points = [self._get_point_coordinates(point_idx) for point_idx in route]
        interpolated_times = []
        
        # Add time for the depot start (0)
        current_time = 0
        
        # Calculate travel time + inspection time for each segment
        inspection_time = self.planner.t_aerial if robot_type == 'aerial' else self.planner.t_ground
        
        for i in range(len(route) - 1):
            prev_idx = route[i]
            curr_idx = route[i+1]
            
            prev_point = points[i]
            curr_point = points[i+1]
            
            # Calculate travel time (distance)
            distance = np.sqrt((prev_point[0] - curr_point[0])**2 + (prev_point[1] - curr_point[1])**2)
            
            # For each interpolation step
            for step in range(20):  # Must match num_steps in _interpolate_path
                # Linear interpolation of time
                step_time = current_time + (step / 20) * distance
                interpolated_times.append(step_time)
            
            # Update current time after travel
            current_time += distance
            
            # Add inspection time if this is a waypoint (not returning to depot)
            if i < len(route) - 2:  # Not the last segment (to depot)
                if curr_idx in times:
                    # Use the completion time from the solution
                    inspection_end_time = times[curr_idx]
                    # Ensure we don't go backward in time
                    if inspection_end_time > current_time:
                        current_time = inspection_end_time
                    else:
                        current_time += inspection_time
                else:
                    current_time += inspection_time
        
        return interpolated_times
    
    def create_animation(self, fps=10, steps_per_segment=20, save_path=None):
        """
        Create the animation of robot paths.
        
        Args:
            fps: Frames per second for the animation
            steps_per_segment: Number of steps to interpolate between waypoints
            save_path: Optional path to save the animation as MP4
            
        Returns:
            Matplotlib animation object
        """
        if not self.planner.solution:
            print("No solution available. Run the solver first.")
            return None
        
        # Prepare the visualization
        self.prepare_visualization()
        
        # Get routes
        aerial_route = self.planner.aerial_route if hasattr(self.planner, 'aerial_route') else []
        ground_route = self.planner.ground_route if hasattr(self.planner, 'ground_route') else []
        
        if not aerial_route or not ground_route:
            print("No routes available. Check the planner solution.")
            return None
        
        # Interpolate paths for smooth animation
        aerial_path = self._interpolate_path(aerial_route, steps_per_segment)
        ground_path = self._interpolate_path(ground_route, steps_per_segment)
        
        # Calculate times for each position
        aerial_times = self._calculate_times(aerial_route, self.planner.aerial_times, 'aerial')
        ground_times = self._calculate_times(ground_route, self.planner.ground_times, 'ground')
        
        # Determine the max time to synchronize animations
        max_time = max(aerial_times[-1] if aerial_times else 0, 
                      ground_times[-1] if ground_times else 0)
        
        # Generate time series for the entire animation
        num_frames = min(int(max_time * fps), 500)  # Limit max frames for performance
        
        # Create animation elements
        aerial_robot = Circle((0, 0), 5, color=self.aerial_color, zorder=10)
        ground_robot = Circle((0, 0), 5, color=self.ground_color, zorder=10)
        
        self.ax.add_patch(aerial_robot)
        self.ax.add_patch(ground_robot)
        
        # Empty lines for paths
        aerial_path_line, = self.ax.plot([], [], color=self.aerial_color, linewidth=2, zorder=5)
        ground_path_line, = self.ax.plot([], [], color=self.ground_color, linewidth=2, zorder=5)
        
        # Time text display
        time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=12, 
                                bbox=dict(facecolor='white', alpha=0.7))
        
        # Track visited waypoints for coloring
        visited_aerial = set()
        visited_ground = set()
        waypoint_patches = {}
        
        # Pre-create waypoint patches
        for i, waypoint in enumerate(self.planner.waypoints):
            patch = self.ax.plot(waypoint[0], waypoint[1], 'o', color=self.waypoint_color, 
                               markersize=8, zorder=7)[0]
            waypoint_patches[i] = patch
        
        def init():
            # Initialize positions
            aerial_robot.center = self._get_point_coordinates(aerial_route[0])
            ground_robot.center = self._get_point_coordinates(ground_route[0])
            
            # Clear paths
            aerial_path_line.set_data([], [])
            ground_path_line.set_data([], [])
            
            # Reset time text
            time_text.set_text('')
            
            # Reset waypoint colors
            for i, patch in waypoint_patches.items():
                patch.set_color(self.waypoint_color)
            
            return [aerial_robot, ground_robot, aerial_path_line, ground_path_line, 
                   time_text] + list(waypoint_patches.values())
        
        def update(frame):
            # Calculate current time
            t = frame / num_frames * max_time
            
            # Update paths and robot positions
            # For aerial robot
            aerial_idx = 0
            for i, time_val in enumerate(aerial_times):
                if time_val <= t:
                    aerial_idx = i
                else:
                    break
            
            # Limit to valid indices
            aerial_idx = min(aerial_idx, len(aerial_path) - 1)
            
            # For ground robot
            ground_idx = 0
            for i, time_val in enumerate(ground_times):
                if time_val <= t:
                    ground_idx = i
                else:
                    break
            
            # Limit to valid indices
            ground_idx = min(ground_idx, len(ground_path) - 1)
            
            # Set robot positions
            if aerial_idx < len(aerial_path):
                aerial_robot.center = (aerial_path[aerial_idx][0], aerial_path[aerial_idx][1])
            
            if ground_idx < len(ground_path):
                ground_robot.center = (ground_path[ground_idx][0], ground_path[ground_idx][1])
            
            # Update path lines
            # Get current path segments
            aerial_x = [p[0] for p in aerial_path[:aerial_idx+1]]
            aerial_y = [p[1] for p in aerial_path[:aerial_idx+1]]
            
            ground_x = [p[0] for p in ground_path[:ground_idx+1]]
            ground_y = [p[1] for p in ground_path[:ground_idx+1]]
            
            aerial_path_line.set_data(aerial_x, aerial_y)
            ground_path_line.set_data(ground_x, ground_y)
            
            # Update time display
            time_text.set_text(f'Time: {t:.1f}')
            
            # Update visited waypoints
            # Calculate which waypoints have been visited
            step_size = steps_per_segment
            
            # For aerial robot
            segment_idx = aerial_idx // step_size
            if segment_idx < len(aerial_route) - 1:
                waypoint_idx = aerial_route[segment_idx]
                if waypoint_idx < self.planner.num_waypoints:
                    visited_aerial.add(waypoint_idx)
            
            # For ground robot
            segment_idx = ground_idx // step_size
            if segment_idx < len(ground_route) - 1:
                waypoint_idx = ground_route[segment_idx]
                if waypoint_idx < self.planner.num_waypoints:
                    visited_ground.add(waypoint_idx)
            
            # Update waypoint colors
            for i, patch in waypoint_patches.items():
                if i in visited_aerial and i in visited_ground:
                    patch.set_color(self.visited_both_color)
                elif i in visited_aerial:
                    patch.set_color(self.visited_aerial_color)
                elif i in visited_ground:
                    patch.set_color(self.visited_ground_color)
            
            return [aerial_robot, ground_robot, aerial_path_line, ground_path_line, 
                   time_text] + list(waypoint_patches.values())
        
        # Create the animation with optimized settings
        self.animation = animation.FuncAnimation(self.fig, update, init_func=init,
                                               frames=num_frames, blit=True, 
                                               interval=1000/fps)
        
        # Save if path is provided
        if save_path:
            # Create directory if needed
            dir_name = os.path.dirname(save_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            
            print(f"Saving animation to {save_path}...")
            
            # Use a more efficient writer with lower bitrate
            writer = animation.FFMpegWriter(
                fps=fps,
                metadata=dict(title='Multi-Robot Inspection', artist='Planner'),
                bitrate=1000,  # Lower bitrate for faster saving
                codec='h264',  # Efficient codec
                extra_args=['-pix_fmt', 'yuv420p']  # For compatibility
            )
            
            # Progress indicator
            self.animation.save(
                save_path, 
                writer=writer,
                progress_callback=lambda i, n: print(f"Saving frame {i+1}/{n}", end="\r")
            )
            print(f"\nAnimation saved to {save_path}")
        
        return self.animation
    
    def show(self):
        """Display the animation."""
        plt.show()
    
    def save_final_paths_image(self, save_path):
        """Save a static image of the final paths without animation."""
        self.prepare_visualization()
        
        # Get routes
        aerial_route = self.planner.aerial_route
        ground_route = self.planner.ground_route
        
        # Plot aerial path
        aerial_coords = [self._get_point_coordinates(point_idx) for point_idx in aerial_route]
        aerial_x = [p[0] for p in aerial_coords]
        aerial_y = [p[1] for p in aerial_coords]
        self.ax.plot(aerial_x, aerial_y, '-', color=self.aerial_color, linewidth=2)
        
        # Plot ground path
        ground_coords = [self._get_point_coordinates(point_idx) for point_idx in ground_route]
        ground_x = [p[0] for p in ground_coords]
        ground_y = [p[1] for p in ground_coords]
        self.ax.plot(ground_x, ground_y, '-', color=self.ground_color, linewidth=2)
        
        # Color waypoints based on visitation
        visited_aerial = set(aerial_route) - {self.planner.num_waypoints}
        visited_ground = set(ground_route) - {self.planner.num_waypoints + 1}
        
        for i, waypoint in enumerate(self.planner.waypoints):
            if i in visited_aerial and i in visited_ground:
                self.ax.plot(waypoint[0], waypoint[1], 'o', color=self.visited_both_color, markersize=8)
            elif i in visited_aerial:
                self.ax.plot(waypoint[0], waypoint[1], 'o', color=self.visited_aerial_color, markersize=8)
            elif i in visited_ground:
                self.ax.plot(waypoint[0], waypoint[1], 'o', color=self.visited_ground_color, markersize=8)

        # Create directory if needed
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Save with higher quality settings
        self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Final paths image saved to {save_path}")
        
# Example usage
if __name__ == "__main__":
    
    # Create and solve the planner
    planner = MultiRobotInspectionPlanner(map_size=(100, 100), num_waypoints=10, seed=42)
    success = planner.solve()
    
    if success:
        # Create visualizer
        visualizer = PathVisualizer(planner)
        
        # Create and display animation
        animation = visualizer.create_animation(fps=30, save_path="animation.mp4")
        
        # Save static image
        visualizer.save_final_paths_image("final_paths.png")
    else:
        print("Failed to find an optimal solution.")