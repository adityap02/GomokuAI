# AI Gomoku: Strategic Play with Convolutional Neural Networks

## Benchmarking CNN's performance with Minimax Algorithm for 15x15 Gomoku Board Game

## Project Overview
This project presents an artificial intelligence (AI) designed to master the game of Gomoku. Using advanced deep learning techniques, we have constructed a Convolutional Neural Network (CNN) that not only learns the intricacies of the game but also develops innovative strategies to compete at high levels of play. This project encapsulates the culmination of research, experimentation, and implementation, providing a window into the future of AI in strategic game theory.

## Comprehensive Methodology
Utilizing the power of CNNs, our model digests the vast state space of the Gomoku board, applying layers of computational neurons to distill complex patterns and strategies from raw data. The model's architecture is engineered to balance computational efficiency with predictive power, resulting in an AI that can adapt to and counter human-like strategic play.

### Detailed Architecture
Our CNN model is composed of:
- Five input layers to encapsulate various game states.
- A 7x7 filter size to capture the breadth of the game board.
- ReLU activation functions to introduce non-linearity and enable complex pattern recognition.
- A Sigmoid output layer to deliver a probabilistic interpretation of the AI's move choices.

## Codebase Structure
Each component of the code serves a specific purpose in bringing the AI to life:

- `submission.py`: Implements the AI's decision-making logic using the trained CNN model.
- `performance.py`: Benchmarks the AI's performance, providing analytics on win rates and strategic efficacy.
- `compete.py`: Facilitates AI vs AI matches, enabling continuous improvement through competition.
- `gomoku.py`: Defines the rules and mechanics of Gomoku, serving as the game's foundation.
- `minimax.py`: A traditional AI algorithm for comparison, demonstrating the CNN's superior strategic depth.
- `training.py`: Manages the training pipeline for the CNN, evolving the AI's capabilities over time.
- `create_dataset.py`: Generates the datasets necessary for effective model training, simulating countless game scenarios.

## Prerequisites
- Python3
- TensorFlow
- sckit-learn

##Dataset

Collected data for training a CNN from the Gomocup website. Utilized standard game data..

