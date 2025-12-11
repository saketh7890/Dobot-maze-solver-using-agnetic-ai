# Dobot-maze-solver-using-agnetic-ai
This project integrates computer vision, pathfinding AI, and robotic control to autonomously navigate and draw paths through a physical maze. Using a camera and the Dobot robotic arm, the system acts as an intelligent agent that perceives, plans, and acts—executing end-to-end maze solving with no human intervention beyond setup.
Key Features

Vision-Based Maze Detection:
Automatically detects and deskews the maze from a live camera feed using OpenCV. Grid lines, walls, and maze boundaries are identified dynamically through adaptive thresholding and line detection.

Color-Based Goal Identification:
Locates red (start) and green (goal) markers within the maze using color segmentation in HSV space.

AI Path Planning:
Implements agentic pathfinder (perplexity_astar_agent) to compute the optimal wall-safe route from start to goal based on grid adjacency graphs.

Homography Calibration:
Transforms pixel coordinates to real-world millimeter space using homography, ensuring accurate robot movements even if the camera is rotated or tilted.

Robotic Execution (Dobot Magician):
Converts the AI-generated path into Dobot coordinates and commands the arm to draw the path physically on the maze using precise motion control (SAFE_Z and DRAW_Z levels).

Technical Stack

Computer Vision: OpenCV, NumPy

Pathfinding: A* Algorithm, AI agent variant

Robotics: Dobot SDK / Serial communication

Coordinate Mapping: Homography transformation, pixel-to-mm calibration

Language: Python

Outcome

The system demonstrates autonomous perception, reasoning, and action — key components of an Agentic AI pipeline. It successfully bridges virtual path planning and real-world robotic execution, showing how intelligent agents can understand and interact with their environments.
