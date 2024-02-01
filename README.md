# Jigsaw Puzzle Solver with OpenCV
## Overview
Jigsaw Puzzle Solver is project that uses advanced image processing and computer vision techniques to automate the solving of jigsaw puzzles.
This solution is designed to handle puzzles with specific characteristics.
#### Key Constraints and Considerations:
* **Shapes of puzzle pieces:** The algorithm is optimized for pieces that closely resemble a square shape.
* **Size of pieces:** There is a size threshold for the puzzle pieces. The pieces should be above a certain size to be accurately processed by the algorithm.(Sample pictures are in the project_images folder)

This project is part of a bigger [project](https://github.com/HamzaAlmahrous/jigsaw-genius). 

## Features
* Solve jigsaw puzzles automatically.
* The option to solve the puzzle with or wihtout a hint which is the picture assembled.


## Overview of the algorithm pipeline
* Preprocessing: Identify puzzle pieces by removing the background, and detecting the contours of each piece by using the cv.findContours function from OpenCV.

* Piece Analysis: For each piece, identify the four corners and classify each edge as either a head, a hole, or straight.

* Solving Strategy: Begin the puzzle assembly with edge pieces, leveraging their unique borders for easier placement. Subsequently, solve for the inner pieces by considering multiple factors: the geometric type of the edges, color correlation between matching piece pairs, and, when a hint image is provided, the similarity of SIFT features between a puzzle piece and the corresponding area in the hint image to ensure accurate matches based on physical fit and visual similarity.


(A detailed explanation of the algorithm is embedded within the provided code.)

## Setup
To run the Jigsaw Puzzle Solver on your local machine, you'll need to have Python and OpenCV installed. Python is the programming language used, and OpenCV is a critical library for image processing functionalities in the project. After setting up these prerequisites, you can proceed to clone the repository onto your local device. Once cloned, navigate to the project directory and run the application.

## Results

### Without hint
![image](https://github.com/Alya0/Jigsaw-Puzzle-Solver/assets/38858460/77fce2b1-1b54-430c-bcf8-f63374eb5849)
![image](https://github.com/Alya0/Jigsaw-Puzzle-Solver/assets/38858460/d5fd9e7e-b7c4-4105-8cd9-a1a91edb4d79)
![image](https://github.com/Alya0/Jigsaw-Puzzle-Solver/assets/38858460/77746cef-3daf-4c3d-a7aa-68cc6d6ea276)


### With hint
![image](https://github.com/Alya0/Jigsaw-Puzzle-Solver/assets/38858460/1a177b97-408f-4893-ad94-9d6882addaea)
![image](https://github.com/Alya0/Jigsaw-Puzzle-Solver/assets/38858460/00910789-4b7b-44ec-8b5d-66e2312b6b4d)

## Refrences
* Using Computer Vision to Solve Jigsaw Puzzles, Travis V. Allen, Standford University.

* https://towardsdatascience.com/solving-jigsaw-puzzles-with-python-and-opencv-d775ba730660

* https://www.youtube.com/watch?v=Bo0RSxt5ECI

