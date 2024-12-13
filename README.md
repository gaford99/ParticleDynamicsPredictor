# Particle Classification using Deep Learning

## A Physics-Informed Neural Network for High Energy Physics
### Project Overview
This project implements a deep learning solution for classifying particle trajectories from high-energy physics collisions. Using data from CERN, the model classifies five different particle types based on their detector signatures.

### Data Description
Source: CERN collision data
Format: 350 pickle files, each representing a unique collision
Structure per file:
Column 1: 10x10 detector images
Column 2: Particle type labels
Sample size: ~3,000 labeled images per collision
### Particle Classes:
11   -> electron
13   -> muon
211  -> pion
321  -> kaon
2212 -> proton
## Technical Architecture
### Model Structure

Python

Collapse
ParticlePhysicsModel
├── Input Layer (10x10 detector images)
├── Physics-Based Feature Extraction
│   ├── Track Features Branch
│   │   └── Curvature and trajectory patterns
│   ├── Energy Features Branch
│   │   └── Energy deposition patterns
│   └── Interaction Features Branch
│       └── Combined physics characteristics
├── Physics Metrics Calculator
│   ├── Track Length
│   ├── Energy Deposition
│   ├── Track Curvature
│   └── Track Density
└── Classification Head
    └── 5 particle types

## Key Components
### Data Processing

Python

def prepare_data(folder_path, batch_size=1024):
    """
    - Loads pickle files
    - Processes 10x10 detector images
    - Creates train/validation split
    - Returns DataLoaders
    """
###Physics-Based Feature Extraction

Python

class PhysicsMetrics:
    """
    Calculates physics-informed features:
    - Track length: Non-zero elements in detector
    - Energy deposition: Sum of detector values
    - Track curvature: Gradient-based calculation
    - Track density: Local hit density analysis
    """
###Training with Physics ConstraintsPython

def train_with_physics_constraints():
    """
    Implements physics-informed training:
    - Classification loss
    - Physics-based regularization
    - Particle-specific constraints
    """
### Class Distribution

Total Samples: 1,176,475
├── Pion (211):   906,047 (77.01%)
├── Kaon (321):   154,323 (13.12%)
├── Proton (2212): 111,730 (9.50%)
├── Electron (11):   3,138 (0.27%)
└── Muon (13):      1,237 (0.11%)

### Performance Metrics

Per-Class Accuracy:
├── Electron: 88.46%
├── Muon:    100.00%
├── Pion:     16.06%
├── Kaon:     17.60%
└── Proton:   32.28%

### Overall Validation Accuracy: 17.13%
Installation and Usage
Requirements
BASH

pip install -r requirements.txt
Requirements include:

PyTorch
NumPy
SciPy
Matplotlib
scikit-learn
Running the Model
Python

# Initialize model
model = ParticlePhysicsModel()

# Prepare data
train_loader, val_loader = prepare_data("path/to/data")

# Train model
training_stats = train_with_physics(model, train_loader, val_loader)

# Visualize results
plot_training_results(training_stats)
Physics-Informed Features
Track Characteristics
Curvature analysis for particle momentum
Density patterns for particle type
Length measurements for energy estimation
Energy Patterns
Deposition intensity
Spatial distribution
Interaction points
Particle-Specific Constraints
Python

# Example physics constraints
Pion:   Medium curvature expected
Kaon:   Higher energy deposition
Proton: Highest energy deposition
Visualization Tools
Feature Maps
Python

visualize_feature_maps():
    """
    Displays:
    - Original detector image
    - Convolutional layer activations
    - Physics-based feature maps
    """
Training Metrics
Python

plot_training_results():
    """
    Shows:
    - Loss curves
    - Accuracy progression
    - Physics violation metrics
    """
Current Limitations and Future Work
Class Imbalance
Dominant pion class (77%)
Very rare muon class (0.11%)
Need for better handling of imbalanced data
Model Performance
Strong on rare classes
Weak on common classes
Room for improvement in overall accuracy
Future Improvements
Enhanced physics constraints
Better data augmentation
Advanced architecture exploration
