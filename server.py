from quart import Quart, render_template, request, Response
import asyncio
import io
import base64
import matplotlib
matplotlib.use('Agg')
import random
from solver import solve_robot_inspection_problem, visualize_solution

app = Quart(__name__)

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/api/optimize', methods=['POST'])
async def optimize():
    # Extract form data
    form = await request.form
    
    # Convert form data to appropriate types
    map_width = float(form.get('map_width', 30))
    map_height = float(form.get('map_height', 30))
    num_waypoints = int(form.get('num_waypoints', 8))
    aerial_speed = float(form.get('aerial_speed', 5.0))
    ground_speed = float(form.get('ground_speed', 2.0))
    aerial_max_time = float(form.get('aerial_max_time', 30.0))
    ground_max_time = float(form.get('ground_max_time', 60.0))
    aerial_inspection_time = float(form.get('aerial_inspection_time', 1.0))
    ground_inspection_time = float(form.get('ground_inspection_time', 3.0))

    
    # Generate random waypoints
    waypoints = [
        (random.uniform(0, map_width), random.uniform(0, map_height))
        for _ in range(num_waypoints)
    ]
    
    # Generate random depot locations
    aerial_depot = (random.uniform(0, map_width), random.uniform(0, map_height))
    ground_depot = (random.uniform(0, map_width), random.uniform(0, map_height))
    
    # Run optimization in a thread pool to not block the event loop
    solution = await asyncio.to_thread(
        solve_robot_inspection_problem,
        waypoints=waypoints,
        aerial_depot=aerial_depot,
        ground_depot=ground_depot,
        aerial_speed=aerial_speed,
        ground_speed=ground_speed,
        aerial_max_time=aerial_max_time,
        ground_max_time=ground_max_time,
        aerial_inspection_time=aerial_inspection_time,
        ground_inspection_time=ground_inspection_time
    )
    
    # Generate visualization image
    if solution["status"] == "optimal":
        # Create image in memory
        img_bytes = io.BytesIO()
        await asyncio.to_thread(
            visualize_solution,
            solution, waypoints, aerial_depot, ground_depot,
            aerial_speed, ground_speed,
            save_path=img_bytes
        )
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    else:
        img_base64 = None
    
    # Return HTML directly instead of JSON
    return await render_template('result_fragment.html', 
                           solution=solution, 
                           img_base64=img_base64)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
