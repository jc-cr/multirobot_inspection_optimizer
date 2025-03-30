import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import io
import base64

def visualize_multi_robot_solution(solution, waypoints, aerial_depot, ground_depot, 
                                 aerial_speed, ground_speed, save_path=None):
    """
    Visualize the solution for multiple aerial and ground robots.
    """
    if solution["status"] != "optimal":
        print("No optimal solution to visualize.")
        return
    
    # Set matplotlib config directory to a temporary directory to avoid permission issues
    temp_dir = tempfile.mkdtemp()
    os.environ['MPLCONFIGDIR'] = temp_dir
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Number of waypoints and robots
    n = len(waypoints)
    k_max = len(solution["aerial_routes"])
    l_max = len(solution["ground_routes"])
    
    # Plot waypoints
    waypoint_x = [waypoints[i][0] for i in range(n)]
    waypoint_y = [waypoints[i][1] for i in range(n)]
    
    # Determine which waypoints are visited by which robots
    waypoint_status = []
    for i in range(n):
        # Check if any aerial robot visited this waypoint
        aerial_visited = any(solution["aerial_visited"][k].get(i, 0) > 0.5 for k in range(k_max))
        # Check if any ground robot visited this waypoint
        ground_visited = any(solution["ground_visited"][l].get(i, 0) > 0.5 for l in range(l_max))
        
        if ground_visited:
            waypoint_status.append('both')
        elif aerial_visited:
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
    
    # Generate colors for multiple robots
    aerial_colors = cm.Blues(np.linspace(0.5, 0.9, k_max))
    ground_colors = cm.Reds(np.linspace(0.5, 0.9, l_max))
    
    # Plot aerial robot routes
    aerial_depot_idx = -1
    legend_elements = []
    
    for k in range(k_max):
        aerial_route = solution["aerial_routes"][k]
        if len(aerial_route) <= 2:  # Only depot-depot, no waypoints
            continue
            
        aerial_route_coords = []
        for i in aerial_route:
            if i == aerial_depot_idx:
                aerial_route_coords.append(aerial_depot)
            else:
                aerial_route_coords.append(waypoints[i])
                
        aerial_x = [coord[0] for coord in aerial_route_coords]
        aerial_y = [coord[1] for coord in aerial_route_coords]
        
        # Get color for this aerial robot
        color = aerial_colors[k]
        
        # Plot route
        ax.plot(aerial_x, aerial_y, '-', color=color, linewidth=2, alpha=0.7, zorder=2)
        
        # Add to legend
        legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                      label=f'Aerial Robot {k} (Speed: {aerial_speed} units/min)'))
        
        # Add arrows and time labels
        for i in range(len(aerial_x) - 1):
            # Add arrow
            ax.annotate('', xy=(aerial_x[i+1], aerial_y[i+1]), 
                        xytext=(aerial_x[i], aerial_y[i]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))
            
            # Add time label for waypoints (not depots)
            idx = aerial_route[i]
            if idx != aerial_depot_idx and i > 0 and idx in solution["aerial_times"][k]:
                time_val = solution["aerial_times"][k][idx]
                midpoint_x = (aerial_x[i] + aerial_x[i-1]) / 2
                midpoint_y = (aerial_y[i] + aerial_y[i-1]) / 2
                ax.text(midpoint_x, midpoint_y, f"{time_val:.1f}m", 
                        color=color, fontsize=8, ha='center', va='bottom')
    
    # Plot ground robot routes
    ground_depot_idx = -2
    
    for l in range(l_max):
        ground_route = solution["ground_routes"][l]
        if len(ground_route) <= 2:  # Only depot-depot, no waypoints
            continue
            
        ground_route_coords = []
        for i in ground_route:
            if i == ground_depot_idx:
                ground_route_coords.append(ground_depot)
            else:
                ground_route_coords.append(waypoints[i])
                
        ground_x = [coord[0] for coord in ground_route_coords]
        ground_y = [coord[1] for coord in ground_route_coords]
        
        # Get color for this ground robot
        color = ground_colors[l]
        
        # Plot route
        ax.plot(ground_x, ground_y, '-', color=color, linewidth=2, alpha=0.7, zorder=2)
        
        # Add to legend
        legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                      label=f'Ground Robot {l} (Speed: {ground_speed} units/min)'))
        
        # Add arrows and time labels
        for i in range(len(ground_x) - 1):
            # Add arrow
            ax.annotate('', xy=(ground_x[i+1], ground_y[i+1]), 
                        xytext=(ground_x[i], ground_y[i]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))
            
            # Add time label for waypoints (not depots)
            idx = ground_route[i]
            if idx != ground_depot_idx and i > 0 and idx in solution["ground_times"][l]:
                time_val = solution["ground_times"][l][idx]
                midpoint_x = (ground_x[i] + ground_x[i-1]) / 2
                midpoint_y = (ground_y[i] + ground_y[i-1]) / 2
                ax.text(midpoint_x, midpoint_y, f"{time_val:.1f}m", 
                        color=color, fontsize=8, ha='center', va='top')
    
    # Add depot and waypoint status to legend
    depot_legend = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
               markersize=10, label='Aerial Depot'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=10, label='Ground Depot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=10, label='Visited by Ground Robot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=10, label='Visited by Aerial Robot Only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
               markersize=10, label='Not Visited')
    ]
    
    legend_elements.extend(depot_legend)
    ax.legend(handles=legend_elements, loc='best', fontsize='small')
    
    # Add solution summary text
    total_waypoints_inspected = solution['objective_value']
    
    summary_text = [
        f"Objective: {total_waypoints_inspected} waypoints inspected"
    ]
    
    for k in range(k_max):
        if k in solution["total_aerial_times"] and solution["total_aerial_times"][k] > 0:
            summary_text.append(f"Aerial {k} time: {solution['total_aerial_times'][k]:.1f} min, dist: {solution['aerial_distances'][k]:.1f}")
    
    for l in range(l_max):
        if l in solution["total_ground_times"] and solution["total_ground_times"][l] > 0:
            summary_text.append(f"Ground {l} time: {solution['total_ground_times'][l]:.1f} min, dist: {solution['ground_distances'][l]:.1f}")
    
    ax.text(0.02, 0.02, "\n".join(summary_text), transform=ax.transAxes, fontsize=9,
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
        plt.close(fig)
    else:
        plt.show()


def generate_time_based_animation(solution, waypoints, aerial_depot, ground_depot, num_extra_frames=5):
    """
    Generates a time-accurate animation showing robots visiting waypoints in correct temporal order.
    Returns a list of base64-encoded PNG images.
    """
    # Set up temporary directory for matplotlib
    temp_dir = tempfile.mkdtemp()
    os.environ['MPLCONFIGDIR'] = temp_dir
    
    # Extract routes and times
    aerial_routes = solution.get("aerial_routes", {})
    ground_routes = solution.get("ground_routes", {})
    aerial_times = solution.get("aerial_times", {})
    ground_times = solution.get("ground_times", {})
    
    # Collect all events (when a robot visits a waypoint)
    events = []
    
    # Add aerial robot visits
    for k in aerial_times:
        for i in aerial_times[k]:
            events.append({
                'time': aerial_times[k][i],
                'type': 'aerial',
                'robot': k,
                'waypoint': i
            })
    
    # Add ground robot visits
    for l in ground_times:
        for i in ground_times[l]:
            events.append({
                'time': ground_times[l][i],
                'type': 'ground',
                'robot': l,
                'waypoint': i
            })
    
    # Sort events by time
    events.sort(key=lambda e: e['time'])
    
    # If we don't have events, return empty list
    if not events:
        return []
    
    # Get min and max times
    min_time = events[0]['time'] if events else 0
    max_time = events[-1]['time'] if events else 0
    
    # Create list of times for frames
    # Start with all event times
    frame_times = [e['time'] for e in events]
    
    # Add some frames at the beginning (time 0)
    frame_times.insert(0, 0)
    
    # Add some frames at the end to show final state
    if max_time > min_time:
        for i in range(1, num_extra_frames + 1):
            buffer_time = max_time * (1 + i * 0.05)  # Add 5% increments
            frame_times.append(buffer_time)
    
    # Remove duplicates and sort
    frame_times = sorted(list(set(frame_times)))
    
    # Generate colors for robots
    k_max = max(aerial_routes.keys()) + 1 if aerial_routes else 0
    l_max = max(ground_routes.keys()) + 1 if ground_routes else 0
    aerial_colors = cm.Blues(np.linspace(0.5, 0.9, max(k_max, 1)))
    ground_colors = cm.Reds(np.linspace(0.5, 0.9, max(l_max, 1)))
    
    # List for storing frame images
    frames = []
    
    # For each timestamp, create a frame
    for current_time in frame_times:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determine waypoint status at this time
        waypoint_status = {}
        waypoint_visited_by = {}
        
        # Initialize all waypoints as 'none' (not visited)
        for i in range(len(waypoints)):
            waypoint_status[i] = 'none'
            waypoint_visited_by[i] = {'aerial': None, 'ground': None}
        
        # Process events up to current time
        for event in events:
            if event['time'] <= current_time:
                waypoint = event['waypoint']
                robot_type = event['type']
                robot_id = event['robot']
                
                # Only update if this is a valid waypoint index
                if 0 <= waypoint < len(waypoints):
                    if robot_type == 'aerial':
                        waypoint_visited_by[waypoint]['aerial'] = robot_id
                        if waypoint_status[waypoint] == 'none':
                            waypoint_status[waypoint] = 'aerial'
                    elif robot_type == 'ground':
                        waypoint_visited_by[waypoint]['ground'] = robot_id
                        # If ground robot visited, it's 'both' regardless of previous state
                        waypoint_status[waypoint] = 'both'
        
        # Plot waypoints with colors based on visit status
        for i, wp in enumerate(waypoints):
            if waypoint_status[i] == 'both':
                ax.scatter(wp[0], wp[1], s=100, color='green', edgecolor='black', zorder=3)
            elif waypoint_status[i] == 'aerial':
                ax.scatter(wp[0], wp[1], s=100, color='lightblue', edgecolor='black', zorder=3)
            else:
                ax.scatter(wp[0], wp[1], s=100, color='lightgray', edgecolor='black', zorder=3)
            
            # Add waypoint label
            ax.text(wp[0], wp[1] + 0.5, f"Waypoint {i}", ha='center', fontsize=9)
            
            # If waypoint has been visited, show which robots visited it
            visited_text = []
            if waypoint_visited_by[i]['aerial'] is not None:
                visited_text.append(f"A{waypoint_visited_by[i]['aerial']}")
            if waypoint_visited_by[i]['ground'] is not None:
                visited_text.append(f"G{waypoint_visited_by[i]['ground']}")
                
            if visited_text:
                ax.text(wp[0], wp[1] - 0.7, ", ".join(visited_text), ha='center', fontsize=8)
        
        # Plot depots
        ax.scatter(aerial_depot[0], aerial_depot[1], s=150, color='blue', 
                  marker='^', edgecolor='black', zorder=4)
        ax.text(aerial_depot[0], aerial_depot[1] + 0.5, "Aerial Depot", ha='center', fontsize=9)
        
        ax.scatter(ground_depot[0], ground_depot[1], s=150, color='red', 
                  marker='s', edgecolor='black', zorder=4)
        ax.text(ground_depot[0], ground_depot[1] + 0.5, "Ground Depot", ha='center', fontsize=9)
        
        # Calculate robot positions at current time
        # Plot aerial robots
        for k in aerial_routes:
            route = aerial_routes[k]
            if len(route) <= 2:  # Skip empty routes
                continue
            
            # Find current position of robot based on time
            robot_position = None
            
            # Convert route indices to positions
            route_positions = []
            for idx in route:
                if idx == -1:  # Aerial depot
                    route_positions.append(aerial_depot)
                elif idx == -2:  # Ground depot
                    route_positions.append(ground_depot)
                else:  # Regular waypoint
                    route_positions.append(waypoints[idx])
            
            # Extract waypoint indices (excluding depots at beginning/end)
            waypoint_indices = route[1:-1]
            
            # Create timeline of when robot reaches each waypoint
            timeline = []
            for i, wp_idx in enumerate(waypoint_indices):
                if wp_idx in aerial_times.get(k, {}):
                    arrival_time = aerial_times[k][wp_idx]
                    timeline.append({
                        'index': i + 1,  # +1 for depot at start
                        'time': arrival_time,
                        'position': route_positions[i + 1]
                    })
            
            # Add depot at beginning and end
            if timeline:
                # Assume time 0 at starting depot
                timeline.insert(0, {
                    'index': 0,
                    'time': 0,
                    'position': route_positions[0]
                })
                
                # Add ending depot using max time from this robot
                max_robot_time = max([t['time'] for t in timeline])
                timeline.append({
                    'index': len(route_positions) - 1,
                    'time': max_robot_time * 1.2,  # Just add some buffer
                    'position': route_positions[-1]
                })
            
            # If we have a timeline, interpolate the robot's position
            if timeline:
                # Sort by time
                timeline.sort(key=lambda x: x['time'])
                
                # Find the time segments that current_time falls between
                for i in range(len(timeline) - 1):
                    t1 = timeline[i]['time']
                    t2 = timeline[i+1]['time']
                    
                    if t1 <= current_time <= t2:
                        # Interpolate position
                        pos1 = timeline[i]['position']
                        pos2 = timeline[i+1]['position']
                        
                        # Linear interpolation
                        if t2 > t1:  # Avoid division by zero
                            fraction = (current_time - t1) / (t2 - t1)
                            
                            robot_x = pos1[0] + fraction * (pos2[0] - pos1[0])
                            robot_y = pos1[1] + fraction * (pos2[1] - pos1[1])
                            
                            robot_position = (robot_x, robot_y)
                            break
                        else:
                            robot_position = pos1
                            break
                
                # If we're after the last waypoint
                if current_time > timeline[-1]['time']:
                    robot_position = timeline[-1]['position']
                
                # If we're before the first waypoint
                if current_time < timeline[0]['time']:
                    robot_position = timeline[0]['position']
            
            # Plot robot if we have a position
            if robot_position:
                color = aerial_colors[k]
                ax.scatter(robot_position[0], robot_position[1], s=120, color=color, edgecolor='black', zorder=5)
                ax.text(robot_position[0], robot_position[1] + 0.3, f"A{k}", ha='center', va='center', 
                       fontsize=8, color='white', fontweight='bold')
        
        # Plot ground robots (similar approach)
        for l in ground_routes:
            route = ground_routes[l]
            if len(route) <= 2:  # Skip empty routes
                continue
            
            # Find current position of robot based on time
            robot_position = None
            
            # Convert route indices to positions
            route_positions = []
            for idx in route:
                if idx == -1:  # Aerial depot
                    route_positions.append(aerial_depot)
                elif idx == -2:  # Ground depot
                    route_positions.append(ground_depot)
                else:  # Regular waypoint
                    route_positions.append(waypoints[idx])
            
            # Extract waypoint indices (excluding depots at beginning/end)
            waypoint_indices = route[1:-1]
            
            # Create timeline of when robot reaches each waypoint
            timeline = []
            for i, wp_idx in enumerate(waypoint_indices):
                if wp_idx in ground_times.get(l, {}):
                    arrival_time = ground_times[l][wp_idx]
                    timeline.append({
                        'index': i + 1,  # +1 for depot at start
                        'time': arrival_time,
                        'position': route_positions[i + 1]
                    })
            
            # Add depot at beginning and end
            if timeline:
                # Assume time 0 at starting depot
                timeline.insert(0, {
                    'index': 0,
                    'time': 0,
                    'position': route_positions[0]
                })
                
                # Add ending depot using max time from this robot
                max_robot_time = max([t['time'] for t in timeline])
                timeline.append({
                    'index': len(route_positions) - 1,
                    'time': max_robot_time * 1.2,  # Just add some buffer
                    'position': route_positions[-1]
                })
            
            # If we have a timeline, interpolate the robot's position
            if timeline:
                # Sort by time
                timeline.sort(key=lambda x: x['time'])
                
                # Find the time segments that current_time falls between
                for i in range(len(timeline) - 1):
                    t1 = timeline[i]['time']
                    t2 = timeline[i+1]['time']
                    
                    if t1 <= current_time <= t2:
                        # Interpolate position
                        pos1 = timeline[i]['position']
                        pos2 = timeline[i+1]['position']
                        
                        # Linear interpolation
                        if t2 > t1:  # Avoid division by zero
                            fraction = (current_time - t1) / (t2 - t1)
                            
                            robot_x = pos1[0] + fraction * (pos2[0] - pos1[0])
                            robot_y = pos1[1] + fraction * (pos2[1] - pos1[1])
                            
                            robot_position = (robot_x, robot_y)
                            break
                        else:
                            robot_position = pos1
                            break
                
                # If we're after the last waypoint
                if current_time > timeline[-1]['time']:
                    robot_position = timeline[-1]['position']
                
                # If we're before the first waypoint
                if current_time < timeline[0]['time']:
                    robot_position = timeline[0]['position']
            
            # Plot robot if we have a position
            if robot_position:
                color = ground_colors[l]
                ax.scatter(robot_position[0], robot_position[1], s=120, color=color, edgecolor='black', zorder=5)
                ax.text(robot_position[0], robot_position[1] + 0.3, f"G{l}", ha='center', va='center', 
                       fontsize=8, color='white', fontweight='bold')
        
        # Add title with current time
        ax.set_title(f'Robot Inspection Plan - Time: {current_time:.1f} minutes', fontsize=14)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set axes labels
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        # Calculate appropriate axis limits
        all_x = [p[0] for p in waypoints] + [aerial_depot[0], ground_depot[0]]
        all_y = [p[1] for p in waypoints] + [aerial_depot[1], ground_depot[1]]
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Add padding
        padding = max(x_max - x_min, y_max - y_min) * 0.2
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        
        # Save the figure to bytes
        plt.tight_layout()
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        
        # Convert to base64
        img_b64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        frames.append(img_b64)
        
        # Close the figure to free memory
        plt.close(fig)
    
    return frames