"""
Menstrual Cycle Prediction REST API
A production-ready FastAPI service for predicting next period start date using deep learning.
Supports both PyTorch and TensorFlow/Keras implementations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Menstrual Cycle Prediction API",
    description="AI-powered menstrual cycle prediction using LSTM/GRU models",
    version="1.0.0"
)

# Enable CORS for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for cycle prediction."""
    past_cycles: List[int] = Field(
        ...,
        description="List of past menstrual cycle lengths in days",
        example=[28, 30, 27, 29, 28, 31, 28, 29, 27, 30, 28, 29]
    )
    last_period_date: str = Field(
        ...,
        description="Last period start date in YYYY-MM-DD format",
        example="2025-01-15"
    )
    framework: Optional[str] = Field(
        default="pytorch",
        description="ML framework to use: 'pytorch' or 'tensorflow'",
        example="pytorch"
    )
    
    @validator('past_cycles')
    def validate_cycles(cls, v):
        if len(v) < 4:
            raise ValueError('Need at least 4 past cycles for prediction')
        if any(c < 20 or c > 45 for c in v):
            raise ValueError('Cycle lengths must be between 20 and 45 days')
        return v
    
    @validator('last_period_date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v
    
    @validator('framework')
    def validate_framework(cls, v):
        if v not in ['pytorch', 'tensorflow']:
            raise ValueError('Framework must be either "pytorch" or "tensorflow"')
        return v

class PredictionResponse(BaseModel):
    """Response model for cycle prediction."""
    predicted_cycle_length: int = Field(..., description="Predicted next cycle length in days")
    predicted_next_period: str = Field(..., description="Predicted next period start date (YYYY-MM-DD)")
    predicted_next_period_formatted: str = Field(..., description="Predicted date in readable format")
    confidence_interval: dict = Field(..., description="Confidence interval for prediction")
    statistics: dict = Field(..., description="Historical cycle statistics")
    uncertainty_days: float = Field(..., description="Prediction uncertainty in days")
    framework_used: str = Field(..., description="ML framework used for prediction")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    timestamp: str

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(cycles, seq_length):
    """
    Prepare time-series data for LSTM/GRU training.
    Creates sequences of past cycles to predict the next cycle.
    """
    if len(cycles) < seq_length + 1:
        seq_length = max(3, len(cycles) - 1)
    
    # Normalize data to [0, 1] range for better training
    cycles_array = np.array(cycles, dtype=np.float32)
    min_val = cycles_array.min()
    max_val = cycles_array.max()
    
    # Avoid division by zero
    if max_val == min_val:
        normalized = np.ones_like(cycles_array) * 0.5
    else:
        normalized = (cycles_array - min_val) / (max_val - min_val)
    
    # Create sequences: [seq1, seq2, ..., seqN] -> next_value
    X, y = [], []
    for i in range(len(normalized) - seq_length):
        X.append(normalized[i:i + seq_length])
        y.append(normalized[i + seq_length])
    
    return np.array(X), np.array(y), min_val, max_val, seq_length

def denormalize(value, min_val, max_val):
    """Convert normalized value back to original scale."""
    if max_val == min_val:
        return min_val
    return value * (max_val - min_val) + min_val

def calculate_uncertainty(cycles):
    """Calculate prediction uncertainty based on historical variance."""
    return np.std(cycles)

# ============================================================================
# PYTORCH IMPLEMENTATION
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    class CycleLSTM(nn.Module):
        """LSTM model for cycle length prediction."""
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super(CycleLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    
    def train_pytorch_model(X, y):
        """Train PyTorch LSTM model."""
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)
        
        model = CycleLSTM(input_size=1, hidden_size=32, num_layers=1)  # Simplified
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        model.train()
        for epoch in range(50):  # Reduced epochs
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        return model
    
    def predict_pytorch(model, last_sequence):
        """Make prediction using trained PyTorch model."""
        model.eval()
        with torch.no_grad():
            last_seq_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).unsqueeze(-1)
            prediction = model(last_seq_tensor)
            return prediction.item()
    
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# ============================================================================
# TENSORFLOW/KERAS IMPLEMENTATION
# ============================================================================

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    def build_keras_model(seq_length):
        """Build Keras LSTM model."""
        model = keras.Sequential([
            layers.LSTM(32, return_sequences=False, input_shape=(seq_length, 1)),  # Simplified
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def train_keras_model(X, y):
        """Train Keras LSTM model."""
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        y_reshaped = y.reshape((y.shape[0], 1))
        
        model = build_keras_model(X.shape[1])
        model.fit(X_reshaped, y_reshaped, epochs=50, batch_size=4, verbose=0)  # Reduced epochs
        
        return model
    
    def predict_keras(model, last_sequence):
        """Make prediction using trained Keras model."""
        last_seq_reshaped = last_sequence.reshape((1, last_sequence.shape[0], 1))
        prediction = model.predict(last_seq_reshaped, verbose=0)
        return prediction[0][0]
    
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ============================================================================
# PREDICTION LOGIC
# ============================================================================

def make_prediction(past_cycles: List[int], last_period_date: str, framework: str):
    """
    Core prediction logic that trains model and generates predictions.
    """
    # Validate framework availability
    if framework == 'pytorch' and not PYTORCH_AVAILABLE:
        raise HTTPException(status_code=500, detail="PyTorch is not installed on the server")
    if framework == 'tensorflow' and not TENSORFLOW_AVAILABLE:
        raise HTTPException(status_code=500, detail="TensorFlow is not installed on the server")
    
    # Preprocess data
    SEQUENCE_LENGTH = 6
    X, y, min_val, max_val, seq_len = preprocess_data(past_cycles, SEQUENCE_LENGTH)
    
    # Train model
    if framework == 'pytorch':
        model = train_pytorch_model(X, y)
    else:
        model = train_keras_model(X, y)
    
    # Prepare last sequence for prediction
    last_sequence = np.array(past_cycles[-seq_len:], dtype=np.float32)
    last_sequence_normalized = (last_sequence - min_val) / (max_val - min_val) if max_val != min_val else np.ones_like(last_sequence) * 0.5
    
    # Make prediction
    if framework == 'pytorch':
        predicted_normalized = predict_pytorch(model, last_sequence_normalized)
    else:
        predicted_normalized = predict_keras(model, last_sequence_normalized)
    
    # Denormalize prediction
    predicted_cycle_length = denormalize(predicted_normalized, min_val, max_val)
    predicted_cycle_length = int(round(predicted_cycle_length))
    
    # Calculate next period date
    last_date = datetime.strptime(last_period_date, "%Y-%m-%d")
    next_period_date = last_date + timedelta(days=predicted_cycle_length)
    
    # Calculate uncertainty
    uncertainty = calculate_uncertainty(past_cycles)
    earliest_date = next_period_date - timedelta(days=int(uncertainty))
    latest_date = next_period_date + timedelta(days=int(uncertainty))
    
    # Compile response
    return {
        "predicted_cycle_length": predicted_cycle_length,
        "predicted_next_period": next_period_date.strftime('%Y-%m-%d'),
        "predicted_next_period_formatted": next_period_date.strftime('%A, %B %d, %Y'),
        "confidence_interval": {
            "predicted_days": predicted_cycle_length,
            "min_days": predicted_cycle_length - int(uncertainty),
            "max_days": predicted_cycle_length + int(uncertainty),
            "earliest_date": earliest_date.strftime('%Y-%m-%d'),
            "latest_date": latest_date.strftime('%Y-%m-%d')
        },
        "statistics": {
            "average_cycle_length": float(np.mean(past_cycles)),
            "std_deviation": float(np.std(past_cycles)),
            "min_cycle": int(min(past_cycles)),
            "max_cycle": int(max(past_cycles)),
            "total_cycles_analyzed": len(past_cycles)
        },
        "uncertainty_days": float(uncertainty),
        "framework_used": framework
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "status": "online",
        "message": "Menstrual Cycle Prediction API is running",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    frameworks = []
    if PYTORCH_AVAILABLE:
        frameworks.append("pytorch")
    if TENSORFLOW_AVAILABLE:
        frameworks.append("tensorflow")
    
    return {
        "status": "healthy",
        "message": f"API is operational. Available frameworks: {', '.join(frameworks)}",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_cycle(request: PredictionRequest):
    """
    Predict next menstrual cycle start date.
    
    - **past_cycles**: List of past cycle lengths in days (minimum 4 cycles)
    - **last_period_date**: Last period start date in YYYY-MM-DD format
    - **framework**: ML framework to use ('pytorch' or 'tensorflow')
    
    Returns predicted cycle length, next period date, and confidence intervals.
    """
    try:
        result = make_prediction(
            past_cycles=request.past_cycles,
            last_period_date=request.last_period_date,
            framework=request.framework
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/frameworks")
async def list_frameworks():
    """List available ML frameworks."""
    return {
        "available_frameworks": {
            "pytorch": PYTORCH_AVAILABLE,
            "tensorflow": TENSORFLOW_AVAILABLE
        },
        "default": "pytorch" if PYTORCH_AVAILABLE else "tensorflow"
    }

@app.get("/favicon.ico")
async def favicon():
    """Favicon handler to prevent 404 errors."""
    return {"message": "No favicon"}

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Railway, Render, etc.) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    print("=" * 70)
    print("MENSTRUAL CYCLE PREDICTION API")
    print("=" * 70)
    print(f"PyTorch Available: {PYTORCH_AVAILABLE}")
    print(f"TensorFlow Available: {TENSORFLOW_AVAILABLE}")
    print("=" * 70)
    print(f"Starting server on port {port}")
    print(f"API Documentation: http://localhost:{port}/docs")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=port)
