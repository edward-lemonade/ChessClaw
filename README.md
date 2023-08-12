# Chess Robot

This repo captures my journey to build a prototype chess robot, which may be
turned into a commercial product in the future.

## What is it?

The robot should

 * Visually recongize a physical chess board and chess pieces in it
 * Use A.I. to determine the next move
 * Drive a robot arm to physically move the pieces
 * Verbally interact with the opponent at each stage

## The idea and the challenges

There are existing A.I. models that I can use to do the visual recognition of
the board and pieces. Stockfish is an open source A.I. chess engine that can be
used to determine the chess moves. I have a retired VEX robot and some spare
parts that I can use to build the robot arm. Maybe hack one of my father's
1st gen echo dots to do the voice recognition and Text-To-Speech. For the
brain, I will probably just use my laptop for now. Once it is tested
successfully, maybe I will transfer the brain code to a mini PC powered with a
battery. The visual sensor can be as simple as a 1080p web cam.

Everything is easy said than done. I may have to deal with a lot of engineering
challenges. Some of them are captured below.

 * what AI model to use for visual recognition?
 * Do I need to fine-tune or train my own AI model?
 * How to convert the recognized board and pieces into something Stockfish can take?
 * What API does Stockfish provide to generate moves?
 * How to connect the VEX robot arm, VEX brain and my laptop?
 * What would the robot arm look like?
 * How to calculate the arm position and control the movement with visual? 

## Design for visual chessboard recognition

### Components

**Notes**:

Pivoted from using **scipy.cluster.hierachy** to 2 level **1D k-means** of
**fixed** number grouping. It is simplified that the result directly form the
board grid. The end result is very good.

 * Visual intake: use opencv-python library
 * Board detection:
   * use canny-edge and hughline algorithm to find the potential grid lines
   * calculate all the intersections
   * use 1d k-means to group the intersections into 11 rows
   * for each role use 1d k-means to group the intersections into 11 columns
   * the average of the group coordinates form rows and columns of the grid
 * Chess piece recognition - Fine tune pre-trained VGG16
   * VGG16 is a pre-trained neural network model does image labeling
   * We need to fine-tune the model to recognize each piece and blank spaces
   * Modifying test.py to generate dataset for training
   * Once board is detected, each cell is cropped out from the image and manually labeled

### Test and dataset capture

```
python3 -m venv .
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test.py
```
You can follow the prompt in the capturing screen to adjust the parameters with
hot keys such as w/s for changing theta ratio, [] for minLen etc. Press
**space** to capture the frame and start cropping images for each cell. Repeat
until **q** is pressed.

![Canny Edge Sample][2] ![hugh_line][3] ![K-Means Gird][4]

### My setup
![My setup][1]

[1]: images/my_setup.jpg "My Setup"
[2]: images/sample_canny_edge.jpg
[3]: images/hough_line.jpg
[4]: images/k-mean-grid.jpg
