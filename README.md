# 🤖 ML Hyperparameter Tuner - OpenEnv

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Train RL agents to optimize ML model hyperparameters using the OpenEnv environment.

## 📋 Overview

This project implements an intelligent hyperparameter tuning system that uses Reinforcement Learning agents to optimize machine learning model configurations. It provides a web-based interface for real-time visualization and monitoring of the optimization process.

**Key Innovation:** Uses OpenEnv to create a custom RL environment where agents learn to select optimal hyperparameters.

## ✨ Features

- 🎯 **RL-Based Optimization** - Train agents to discover optimal hyperparameters
- 🌐 **Web Interface** - Real-time monitoring and visualization dashboard
- 🔧 **Flexible Configuration** - Support for multiple model types and hyperparameters
- 📊 **Performance Tracking** - Track optimization progress and model metrics
- 🐳 **Dockerized** - Easy deployment with Docker containers
- 🚀 **Production Ready** - FastAPI backend with async support

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip or conda
- Docker (optional)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Sanskriti2305/ml-hyperparameter-tuner.git
cd ml-hyperparameter-tuner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn server.app:app --port 8000 --reload
```

Visit: **http://localhost:8000**

### Docker Setup

```bash
docker build -t ml-hyperparameter-tuner .
docker run -p 8000:8000 ml-hyperparameter-tuner
```

## 📁 Project Structure

```
ml-hyperparameter-tuner/
├── server/
│   ├── app.py              # FastAPI application
│   ├── environment.py      # OpenEnv environment setup
│   ├── Dockerfile          # Docker configuration
│   └── requirements.txt     # Backend dependencies
├── src/
│   ├── client.py           # API client utilities
│   ├── models.py           # Data models & schemas
│   └── utils/              # Helper functions
├── inference.py            # Inference module
├── validate_submission.py   # Validation utilities
├── openenv.yaml            # Configuration file
├── pyproject.toml          # Project metadata
└── README.md               # This file
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           Web Interface (Frontend)               │
└────────────────┬────────────────────────────────┘
                 │
         ┌───────▼──────────┐
         │  FastAPI Server  │
         ├──────────────────┤
         │ REST API Routes  │
         └───────┬──────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐  ┌─────▼────┐  ┌───▼────┐
│ RL    │  │  Model   │  │Database│
│Agent  │  │ Training │  │        │
└───────┘  └──────────┘  └────────┘
```

## 🎮 How It Works

1. **Environment Setup** - OpenEnv creates a custom RL environment
2. **Agent Training** - RL agents explore hyperparameter space
3. **Model Evaluation** - Each agent action trains a model and evaluates it
4. **Optimization** - Agents learn to maximize model performance
5. **Visualization** - Real-time tracking via web dashboard

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/status` | Current optimization status |
| POST | `/api/optimize` | Start optimization process |
| GET | `/api/results` | Get optimization results |
| GET | `/api/metrics` | Get performance metrics |

## 🔧 Configuration

Edit `openenv.yaml` to customize:

```yaml
models:
  - type: "neural_network"
    hyperparameters:
      - learning_rate: [0.001, 0.1]
      - batch_size: [16, 128]
      - epochs: [10, 100]

agent:
  type: "ppo"  # Proximal Policy Optimization
  episodes: 100
```

## 📈 Performance Metrics

Monitor optimization performance:
- **Reward Trend** - Agent learning progress
- **Model Accuracy** - Best model performance
- **Hyperparameter Distribution** - Parameter space exploration
- **Training Time** - Efficiency metrics

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (\`git checkout -b feature/amazing-feature\`)
3. Commit your changes (\`git commit -m 'Add amazing feature'\`)
4. Push to the branch (\`git push origin feature/amazing-feature\`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Sanskriti** - [@Sanskriti2305](https://github.com/Sanskriti2305)

## 🔗 Links

- 🌐 **Live Demo:** [HF Spaces](https://huggingface.co/spaces/Violet2305/ml-hyperparameter-tuner)
- 📖 **OpenEnv Docs:** [Documentation](https://github.com/openenv/docs)
- 📚 **RL Concepts:** [Reinforcement Learning Primer](https://spinningup.openai.com/)

##  Acknowledgments

- OpenEnv team for the environment framework
- PyTorch and FastAPI communities
- All contributors and testers

## ⭐ Show Your Support

Give a ⭐ if this project helped you!

---

**Made with ❤️ for ML Optimization**

