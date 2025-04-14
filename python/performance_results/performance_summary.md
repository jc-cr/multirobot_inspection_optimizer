# MILP Solver Performance Analysis Report

## Overview
This report presents the computational performance analysis of the Multi-Robot Inspection MILP solver
as the problem scales in terms of map size and number of waypoints.

## Map Size Scaling
- Tested map sizes: [100, 200, 500, 1000]
- Fixed number of waypoints: 20
- See figures: map_size_vs_time.png

## Waypoint Scaling
- Fixed map size: 500 x 500
- Tested waypoint counts: [5, 10, 15, 20, 25]
- See figures: waypoints_vs_time.png, waypoints_vs_objective.png

## Computational Complexity Analysis
- Map size complexity: Polynomial of degree 3 (R² = 1.0000)
- Waypoint complexity: Polynomial of degree 3 (R² = 0.5154)

## Performance Summary
- Minimum solve time for map size experiment: 4.21 seconds (map size: 100)
- Maximum solve time for map size experiment: 83.73 seconds (map size: 500)
- Minimum solve time for waypoint experiment: 0.08 seconds (waypoints: 5)
- Maximum solve time for waypoint experiment: 17.78 seconds (waypoints: 15)

## Conclusions
The performance analysis reveals how the MILP solver scales with both map size and number of waypoints.
The polynomial fits to the solve time data give insight into the computational complexity class of the problem.
These results can be used to predict performance for larger problem instances and to identify potential
bottlenecks in the solution approach.