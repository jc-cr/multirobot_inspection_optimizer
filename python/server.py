from quart import Quart, render_template, request, Response, jsonify
import asyncio
import io
import base64
import matplotlib
matplotlib.use('Agg')
import random
import json
from solver_code.solver import solve_multi_robot_inspection_problem
from solver_code.figures import visualize_multi_robot_solution, generate_time_based_animation

app = Quart(__name__)

MAP_SIZE = 100

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/api/optimize', methods=['POST'])
async def optimize():
    # Extract form data
    form = await request.form
    
    # Map setting
    map_width = float(form.get('map_width', MAP_SIZE))
    map_height = float(form.get('map_height', MAP_SIZE))
    
    # Parse waypoints
    waypoints_data = form.get('waypoints', '[]')
    try:
        waypoints = json.loads(waypoints_data)
    except json.JSONDecodeError:
        waypoints = []
    
    # Only use the provided waypoints, don't generate random ones if none provided
    if not waypoints:
        print("No waypoints provided, optimization will likely fail")
        # The solver expects at least some waypoints, so provide a single default one
        # but this will likely be suboptimal
        waypoints = [(map_width/2, map_height/2)]
    
    # Robot parameters
    aerial_speed = float(form.get('aerial_speed', 5.0))
    ground_speed = float(form.get('ground_speed', 2.0))
    aerial_max_time = float(form.get('aerial_max_time', 30.0))
    ground_max_time = float(form.get('ground_max_time', 60.0))
    aerial_inspection_time = float(form.get('aerial_inspection_time', 1.0))
    ground_inspection_time = float(form.get('ground_inspection_time', 3.0))
    num_aerial_robots = int(form.get('num_aerial_robots', 1))
    num_ground_robots = int(form.get('num_ground_robots', 1))
    
    # Depot locations
    aerial_depot_x = float(form.get('aerial_depot_x', 0))
    aerial_depot_y = float(form.get('aerial_depot_y', 0))
    ground_depot_x = float(form.get('ground_depot_x', map_width))
    ground_depot_y = float(form.get('ground_depot_y', 0))
    
    aerial_depot = (aerial_depot_x, aerial_depot_y)
    ground_depot = (ground_depot_x, ground_depot_y)
    
    print(f"Starting optimization with {len(waypoints)} waypoints, {num_aerial_robots} aerial robots, {num_ground_robots} ground robots")
    
    # Run optimization in a thread pool to not block the event loop
    solution = await asyncio.to_thread(
        solve_multi_robot_inspection_problem,
        waypoints=waypoints,
        aerial_depot=aerial_depot,
        ground_depot=ground_depot,
        aerial_speed=aerial_speed,
        ground_speed=ground_speed,
        aerial_max_time=aerial_max_time,
        ground_max_time=ground_max_time,
        aerial_inspection_time=aerial_inspection_time,
        ground_inspection_time=ground_inspection_time,
        num_aerial_robots=num_aerial_robots,
        num_ground_robots=num_ground_robots
    )
    
    # Generate both static visualization and animation frames
    if solution["status"] == "optimal":
        # Create image in memory for static visualization
        img_bytes = io.BytesIO()
        await asyncio.to_thread(
            visualize_multi_robot_solution,
            solution, waypoints, aerial_depot, ground_depot,
            aerial_speed, ground_speed,
            save_path=img_bytes
        )
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        
        # Generate animation frames using the simple method
        animation_frames = await asyncio.to_thread(
            generate_time_based_animation,
            solution, waypoints, aerial_depot, ground_depot
        )
    else:
        img_base64 = None
        animation_frames = []
    
    # Return HTML directly instead of JSON
    return await render_template('result_fragment.html', 
                           solution=solution, 
                           img_base64=img_base64,
                           animation_frames=animation_frames)

@app.route('/api/add_random_waypoint', methods=['POST'])
async def add_random_waypoint():
    form = await request.form
    map_width = float(form.get('map_width', MAP_SIZE))
    map_height = float(form.get('map_height', MAP_SIZE))
    
    # Generate a random waypoint
    new_waypoint = (random.uniform(0, map_width), random.uniform(0, map_height))
    
    return {"x": new_waypoint[0], "y": new_waypoint[1]}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)