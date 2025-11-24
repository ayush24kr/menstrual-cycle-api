# 🌸 Menstrual Cycle Prediction API

AI-powered REST API for predicting next menstrual cycle start dates using LSTM/GRU deep learning models. Built with FastAPI and supports both PyTorch and TensorFlow backends.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Features

- ✨ **AI-Powered Predictions** - Uses LSTM neural networks for time-series forecasting
- 🔄 **Dual Framework Support** - Choose between PyTorch or TensorFlow
- 📊 **Statistical Analysis** - Provides confidence intervals and historical statistics
- 🔒 **Production Ready** - Input validation, error handling, and CORS support
- 📖 **Interactive Docs** - Auto-generated Swagger UI documentation
- ⚡ **Fast & Lightweight** - Optimized for serverless deployment

## 📋 Table of Contents

- [Demo](#-demo)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Usage Examples](#-usage-examples)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

## 🎮 Demo

**Live API:** https://pleasant-spontaneity-production-bd9d.up.railway.app

**Interactive Docs:** https://pleasant-spontaneity-production-bd9d.up.railway.app/docs

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- pip or conda

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/menstrual-cycle-api.git
   cd menstrual-cycle-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   
   For PyTorch (recommended):
   ```bash
   pip install -r requirements.txt
   ```
   
   For TensorFlow:
   ```bash
   pip install fastapi uvicorn[standard] pydantic numpy tensorflow-cpu
   ```

## 🚦 Quick Start

### Run Locally

```bash
python menstrual_cycle_predictor.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Make Your First Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "past_cycles": [28, 30, 27, 29, 28, 31, 28, 29, 27, 30, 28, 29],
    "last_period_date": "2025-01-15",
    "framework": "pytorch"
  }'
```

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root health check |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Predict next menstrual cycle |
| `/frameworks` | GET | List available ML frameworks |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

### Request Schema

**POST /predict**

```json
{
  "past_cycles": [28, 30, 27, 29, 28, 31, 28, 29, 27, 30, 28, 29],
  "last_period_date": "2025-01-15",
  "framework": "pytorch"
}
```

**Parameters:**
- `past_cycles` (required): Array of past cycle lengths in days (minimum 4 cycles)
- `last_period_date` (required): Last period start date (YYYY-MM-DD format)
- `framework` (optional): ML framework to use - `"pytorch"` or `"tensorflow"` (default: `"pytorch"`)

**Validation Rules:**
- Minimum 4 past cycles required
- Cycle lengths must be between 20-45 days
- Date must be in YYYY-MM-DD format

### Response Schema

```json
{
  "predicted_cycle_length": 29,
  "predicted_next_period": "2025-02-13",
  "predicted_next_period_formatted": "Thursday, February 13, 2025",
  "confidence_interval": {
    "predicted_days": 29,
    "min_days": 27,
    "max_days": 31,
    "earliest_date": "2025-02-11",
    "latest_date": "2025-02-15"
  },
  "statistics": {
    "average_cycle_length": 28.5,
    "std_deviation": 1.31,
    "min_cycle": 27,
    "max_cycle": 31,
    "total_cycles_analyzed": 12
  },
  "uncertainty_days": 1.31,
  "framework_used": "pytorch"
}
```

## 💡 Usage Examples

### Python

```python
import requests

url = "https://pleasant-spontaneity-production-bd9d.up.railway.app/predict"

data = {
    "past_cycles": [28, 30, 27, 29, 28, 31, 28, 29, 27, 30, 28, 29],
    "last_period_date": "2025-01-15",
    "framework": "pytorch"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Next period predicted: {result['predicted_next_period']}")
print(f"Cycle length: {result['predicted_cycle_length']} days")
print(f"Confidence: ±{result['uncertainty_days']:.1f} days")
```

### JavaScript/Fetch

```javascript
const predictCycle = async () => {
  const response = await fetch('https://pleasant-spontaneity-production-bd9d.up.railway.app/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      past_cycles: [28, 30, 27, 29, 28, 31, 28, 29, 27, 30, 28, 29],
      last_period_date: "2025-01-15",
      framework: "pytorch"
    })
  });
  
  const data = await response.json();
  console.log('Next period:', data.predicted_next_period);
};

predictCycle();
```

### cURL

```bash
# Make prediction
curl -X POST "https://pleasant-spontaneity-production-bd9d.up.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "past_cycles": [28, 30, 27, 29, 28, 31],
    "last_period_date": "2025-01-15",
    "framework": "pytorch"
  }'

# Check health
curl https://pleasant-spontaneity-production-bd9d.up.railway.app/health

# List available frameworks
curl https://pleasant-spontaneity-production-bd9d.up.railway.app/frameworks
```

## 🚀 Deployment

### Deploy to Railway

1. **Create `requirements.txt`:**
   ```txt
   fastapi==0.115.0
   uvicorn[standard]==0.30.0
   pydantic==2.9.0
   numpy==1.26.4
   --extra-index-url https://download.pytorch.org/whl/cpu
   torch==2.1.0+cpu
   ```

2. **Create `Procfile`:**
   ```
   web: uvicorn menstrual_cycle_predictor:app --host 0.0.0.0 --port $PORT
   ```

3. **Create `runtime.txt`:**
   ```
   python-3.11.5
   ```

4. **Deploy:**
   ```bash
   # Push to GitHub
   git add .
   git commit -m "Initial commit"
   git push

   # Deploy on Railway
   # 1. Go to railway.app
   # 2. Create new project from GitHub repo
   # 3. Railway auto-detects and deploys!
   ```

### Deploy to Other Platforms

#### Render
```bash
# Same files as Railway
# Deploy from dashboard at render.com
```

#### Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "menstrual_cycle_predictor:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t cycle-prediction-api .
docker run -p 8000:8000 cycle-prediction-api
```

## ⚙️ Configuration

### Environment Variables

```bash
# Optional: Set default framework
export DEFAULT_FRAMEWORK=pytorch

# Optional: Set port (default: 8000)
export PORT=8080
```

### Model Hyperparameters

Edit in `menstrual_cycle_predictor.py`:

```python
SEQUENCE_LENGTH = 6      # Number of past cycles to use
EPOCHS = 50              # Training epochs
HIDDEN_SIZE = 32         # LSTM hidden layer size
LEARNING_RATE = 0.01     # Optimizer learning rate
```

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run tests
pytest tests/
```

### Manual Testing

Visit the interactive documentation:
```
http://localhost:8000/docs
```

## 🔒 Security Notes

⚠️ **Important Disclaimers:**
- This API is for **informational purposes only**
- Not a substitute for professional medical advice
- Predictions are based on historical patterns
- Individual cycle variations are normal
- Always consult healthcare professionals for medical concerns

## 📊 Model Information

### Architecture
- **Model Type**: LSTM (Long Short-Term Memory)
- **Input**: Sequence of past cycle lengths
- **Output**: Predicted next cycle length
- **Features**: Time-series pattern recognition

### Performance
- **Training Time**: ~100-300ms per request
- **Accuracy**: Varies based on cycle regularity
- **Confidence Interval**: ±1 standard deviation

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linter
flake8 .

# Format code
black .

# Type checking
mypy .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- ML powered by [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/)
- Deployed on [Railway](https://railway.app/)

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/ayush24kr/menstrual-cycle-api/issues)

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Made with ❤️ for better cycle tracking**

[![Star on GitHub](https://img.shields.io/github/stars/yourusername/menstrual-cycle-api.svg?style=social)](https://github.com/yourusername/menstrual-cycle-api)
