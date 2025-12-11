🚀 Dobot Maze Solver Using Agentic AI

This project integrates computer vision, AI-driven pathfinding, and robotic control to autonomously navigate and draw paths through a physical maze. Using a camera and the Dobot robotic arm, the system behaves as an intelligent agent that perceives → plans → acts, executing end-to-end maze solving with zero human intervention after setup.

🔍 Key Features
🧠 Vision-Based Maze Detection

Automatically detects and deskews the maze from a live camera feed using OpenCV.

Identifies grid lines, walls, and boundaries via adaptive thresholding and line detection.

🎯 Color-Based Goal Identification

Detects red (start) and green (goal) markers through HSV color segmentation.

🤖 AI Path Planning

Uses an agentic pathfinder (perplexity_astar_agent) implementing the A* algorithm.

Computes an optimal, wall-safe route from start to goal using a grid-based adjacency graph.

📐 Homography Calibration

Converts pixel coordinates to millimeter space using homography transformation.

Maintains accuracy even when the camera is rotated or misaligned.

🦾 Robotic Execution (Dobot Magician)

Translates AI-planned paths into Dobot coordinates.

Commands the arm to physically draw the solution path using precise movement control with SAFE_Z and DRAW_Z levels.

🛠️ Technical Stack
Component	Technologies
Computer Vision	OpenCV, NumPy
Pathfinding	A* Algorithm, Agentic AI
Robotics	Dobot SDK, Serial Communication
Coordinate Mapping	Homography, Pixel-to-mm Calibration
Language	Python

✅ Outcome

This system showcases an Agentic AI pipeline that fully integrates perception, reasoning, and action.
It successfully bridges digital path planning with real-world robotic execution, demonstrating how intelligent agents can interpret their environment and perform precise physical tasks autonomously.
