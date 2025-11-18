# FlexPINN: Physics-Informed Neural Networks for Mixer Design

A flexible Physics-Informed Neural Network (PINN) framework for optimizing mixer geometries with advanced boundary handling and multi-parameter learning.

## Project Structure

### Core Components:
- **`config.py`** - Configuration constants and device setup
- **`network.py`** - Neural network architectures (Large/Small/Combined networks)
- **`flex_pinn.py`** - Main PINN training class with adaptive weighting
- **`losses.py`** - Physics, boundary condition, and penalty loss functions

### Execution Scripts:
- **`main_pinn.py`** - Main training pipeline
- **`main_bc_baffles.py`** - Boundary condition generation
- **`main_collocations.py`** - Collocation point generation
- **`main_data_aggregation.py`** - Data aggregation and visualization

### Utilities:
- **`constants.py`** - Physical constants and parameters
- **`utils.py`** - Visualization and helper functions

## Key Features
- Multi-network architecture for separate variable learning
- Adaptive loss weighting for balanced training
- Complex geometry handling with cubic spline boundaries
- Comprehensive boundary condition enforcement
- GPU-accelerated PyTorch implementation

