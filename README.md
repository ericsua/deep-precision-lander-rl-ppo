# Deep Precision Lander RL - PPO

A Reinforcement Learning project for training a fuel-efficient and precise Lunar Lander agent using Proximal Policy Optimization (PPO).

## 🎯 Project Overview

This project implements a custom reward function for the `LunarLander-v3` environment that emphasizes:
- **Fuel Efficiency**: Heavily penalizes fuel consumption (2.5x multiplier)
- **Precision Landing**: Rewards landing close to the center between the flags

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd deep-precision-lander-rl-ppo
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

3. **Install development dependencies (optional):**
   ```bash
   uv sync --extra dev
   ```

## 📁 Project Structure

```
deep-precision-lander-rl-ppo/
├── src/
│   └── deep_precision_lander_rl_ppo/
│       ├── __init__.py
│       └── custom_lander_env.py    # Custom environment wrapper
├── scripts/
│   ├── train.py                    # Main training script
│   └── evaluate.py                 # Model evaluation script
├── tests/
│   ├── __init__.py
│   └── test_custom_lander_env.py   # Unit tests
├── pyproject.toml                  # Project configuration
├── .python-version                 # Python version specification
├── Makefile                        # Development tasks
├── README.md                       # This file
├── logs/                           # Training logs and TensorBoard data
└── models/                         # Trained model files
```

## 🔧 Usage

### 1. Training the Agent

Run the training script using uv:

```bash
uv run python scripts/train.py
```

Or use the Makefile:

```bash
make train
```

This will:
- Create a custom `EfficientLanderEnv` environment
- Train a PPO model for 200,000 timesteps
- Save the best model during training to `models/best_model.zip`
- Save the final model to `models/ppo_efficient_lander.zip`
- Log training progress to `logs/` for TensorBoard

### 2. Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

Open your browser to `http://localhost:6006` to see training metrics.

### 3. Evaluating the Trained Agent

Watch your trained agent in action:

```bash
uv run python scripts/evaluate.py
```

Or use the Makefile:

```bash
make evaluate
```

This will:
- Load the best trained model
- Run 10 evaluation episodes
- Render the environment so you can watch the agent
- Display performance metrics

## 🛠️ Development

### Code Quality

The project includes several development tools configured in `pyproject.toml`:

- **Formatting**: Black for code formatting
- **Import Sorting**: isort for import organization
- **Linting**: flake8 for code quality checks
- **Type Checking**: mypy for static type analysis
- **Testing**: pytest for unit testing

### Development Commands

Use the Makefile for common development tasks:

```bash
# Install dependencies
make install

# Install development dependencies
make install-dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Check code formatting
make format-check

# Sort imports
make sort

# Check import sorting
make sort-check

# Run linting
make lint

# Run type checking
make type-check

# Clean build artifacts
make clean

# Run all quality checks
make all
```

Or use uv directly:

```bash
# Run tests
uv run pytest

# Format code
uv run black src tests scripts

# Sort imports
uv run isort src tests scripts

# Run linting
uv run flake8 src tests scripts

# Run type checking
uv run mypy src/deep_precision_lander_rl_ppo
```

## 🧠 Custom Environment Details

### Reward Function Modifications

The `EfficientLanderEnv` wrapper implements custom reward logic:

1. **Fuel Consumption Penalty**:
   - When any engine is fired (actions 1, 2, 3), negative rewards are multiplied by 2.5
   - This heavily discourages unnecessary fuel usage

2. **Precision Landing Bonus**:
   - Upon successful landing, calculates bonus based on horizontal distance from center
   - Perfect center landing: +100 points
   - Bonus decreases exponentially with distance from center

### Action Space
- **0**: No operation
- **1**: Main engine
- **2**: Left engine
- **3**: Right engine

## ⚙️ Model Configuration

The PPO model is configured with:
- **Policy**: Multi-layer Perceptron (`MlpPolicy`)
- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **Training Steps**: 2048 per update
- **Epochs**: 10 per batch
- **Device**: Automatically detects MPS (Apple Silicon), CUDA, or CPU

## 📊 Expected Results

With the custom reward function, you should see:
- Agents that use minimal fuel
- Precise landings near the center
- Higher episode rewards for efficient performance
- Learning curves that show improvement in both fuel efficiency and landing precision

## 🛠️ Customization

### Modifying Reward Weights

Edit `src/deep_precision_lander_rl_ppo/custom_lander_env.py` to adjust:
- `fuel_consumption_penalty`: Change from 2.5 to your preferred multiplier
- Precision bonus calculation in `_calculate_precision_bonus()`

### Training Parameters

Modify `scripts/train.py` to adjust:
- Training timesteps
- Learning rate
- Batch size
- PPO hyperparameters

## 🐛 Troubleshooting

### Common Issues

1. **Box2D Installation**: If you encounter Box2D issues, ensure you have the required system dependencies:
   ```bash
   # On macOS
   brew install swig
   
   # On Ubuntu/Debian
   sudo apt-get install swig
   ```

2. **Rendering Issues**: If the evaluation window doesn't render properly, try:
   ```bash
   export DISPLAY=:0  # On Linux
   ```

3. **Memory Issues**: Reduce batch size or training steps if you encounter memory problems.

4. **Python Version**: The project requires Python 3.9+. Check your version with:
   ```bash
   python --version
   ```

## 📚 Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [LunarLander Environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [uv Documentation](https://docs.astral.sh/uv/)

## 🤝 Contributing

This is a hackathon project! Feel free to:
- Experiment with different reward functions
- Try different RL algorithms
- Optimize hyperparameters
- Add new features
- Improve code quality

## 📄 License

This project is created for educational and hackathon purposes.

---

**Happy Hacking! 🚀**
