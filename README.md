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
parts that I can use to build the robot arm. Maybe tear down one of my father's
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
