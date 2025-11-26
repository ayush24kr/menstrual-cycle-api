"""
Combined Reproductive Health API
Combines both the Chatbot and Menstrual Cycle Predictor into a single API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime, timedelta
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Reproductive Health Combined API",
    description="AI-powered chatbot and menstrual cycle prediction in one API",
    version="2.0.0"
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
# CHATBOT SECTION - GROQ INTEGRATION
# ============================================================================

from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# UPDATED MODEL - Current as of November 2024
MODEL_NAME = "llama-3.3-70b-versatile"  # ⭐ RECOMMENDED - Best performance

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    safety_triggered: Optional[bool] = False

# Safety Keywords
EMERGENCY_KEYWORDS = [
    "severe pain", "heavy bleeding", "can't breathe", "chest pain",
    "unconscious", "seizure", "extremely dizzy", "fainted",
    "severe headache", "vision loss", "severe abdominal pain",
    "hemorrhaging", "extreme bleeding", "can't stop bleeding",
    "pregnancy complication", "severe cramping pregnancy"
]

UNSAFE_KEYWORDS = [
    "self-harm", "suicide", "kill myself", "end my life",
    "perform surgery", "diy surgery", "home surgery",
    "abortion at home", "self-induce", "coat hanger",
    "terminate pregnancy myself", "dangerous pills"
]

# Health-related keywords for topic validation
HEALTH_RELATED_KEYWORDS = [
    # Menstrual cycle
    "period", "menstruation", "menstrual", "cycle", "pms", "pmdd",
    "cramps", "cramping", "bleeding", "spotting", "flow",
    # Reproductive health
    "pregnancy", "pregnant", "ovulation", "fertility", "conception",
    "contraception", "birth control", "iud", "pill",
    # Anatomy
    "uterus", "ovaries", "vagina", "cervix", "fallopian", "endometrium",
    # Hormones
    "estrogen", "progesterone", "hormone", "hormonal",
    # Symptoms
    "discharge", "pain", "irregular", "heavy", "light", "late",
    # General health
    "gynecologist", "obgyn", "doctor", "health", "medical",
    "symptoms", "pcos", "endometriosis", "fibroids",
    # Hygiene
    "tampon", "pad", "menstrual cup", "hygiene",
    # Pregnancy related
    "trimester", "fetus", "baby", "labor", "delivery", "breastfeeding",
    "postpartum", "miscarriage", "abortion"
]

# Off-topic keywords (obviously unrelated topics)
OFF_TOPIC_KEYWORDS = [
    # Technology
    "computer", "laptop", "software", "programming", "code", "python",
    "javascript", "app development", "website", "algorithm",
    # Sports
    "football", "basketball", "cricket", "soccer", "tennis",
    # Entertainment
    "movie", "film", "music", "song", "game", "video game",
    "netflix", "youtube", "tiktok",
    # Food (unless related to health)
    "recipe", "cooking", "restaurant", "pizza", "burger",
    # General unrelated
    "weather", "politics", "stock market", "cryptocurrency",
    "car", "vehicle", "travel destination", "vacation",
    "math homework", "history", "geography"
]

SYSTEM_PROMPT = """You are a supportive, knowledgeable reproductive health education assistant for girls and women. Your role is to provide accurate, general educational information only.

CRITICAL SAFETY RULES YOU MUST FOLLOW:
1. NEVER provide medical diagnosis or diagnostic conclusions
2. NEVER prescribe treatments, medications, or dosages
3. NEVER provide abortion methods or procedures
4. NEVER provide instructions for self-surgery or dangerous home remedies
5. ALWAYS encourage professional medical consultation for specific health concerns
6. Provide only general educational information about reproductive health
7. ONLY answer questions related to reproductive health, menstrual cycles, pregnancy, fertility, and women's health
8. REFUSE to answer questions about unrelated topics like technology, sports, entertainment, food recipes, etc.

TOPICS YOU CAN DISCUSS:
- Menstrual cycle education (phases, normal variations)
- Period pain management basics (heat, rest, OTC options to discuss with doctor)
- Reproductive anatomy education
- Hormones and their general functions
- Pregnancy basics and what to expect
- Fertility basics and ovulation
- Hygiene and general self-care
- When to see a doctor
- PCOS, endometriosis, and other reproductive health conditions (educational info only)

TOPICS YOU CANNOT DISCUSS:
- Technology, programming, computers
- Sports, entertainment, movies, music
- Food recipes (unless directly related to menstrual health)
- Politics, finance, travel
- General knowledge unrelated to reproductive health

YOUR COMMUNICATION STYLE:
- Polite, supportive, and non-judgmental
- Use clear, easy-to-understand language
- Normalize reproductive health discussions
- Validate feelings and concerns
- Keep responses concise (2-3 paragraphs)
- Always end with encouragement to consult healthcare providers

RESPONSE FORMAT:
- Provide educational information
- Include relevant general facts
- Always include a disclaimer: "This is general educational information. For personalized medical advice, please consult a qualified healthcare provider."

Remember: You are an educational resource, not a replacement for medical professionals."""

TOPIC_VALIDATION_PROMPT = """You are a topic classifier. Determine if the following question is related to reproductive health, menstrual cycles, pregnancy, fertility, or women's health.

Respond with ONLY one word:
- "RELEVANT" if the question is about reproductive health, periods, pregnancy, fertility, hormones, gynecological health, or related topics
- "IRRELEVANT" if the question is about technology, sports, entertainment, food, politics, general knowledge, or any other unrelated topic

Question: {message}

Classification:"""

def check_emergency(message: str) -> bool:
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in EMERGENCY_KEYWORDS)

def check_unsafe(message: str) -> bool:
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in UNSAFE_KEYWORDS)

def is_obviously_off_topic(message: str) -> bool:
    """Quick check for obviously off-topic questions using keywords."""
    message_lower = message.lower()
    
    # Check for off-topic keywords
    has_off_topic = any(keyword in message_lower for keyword in OFF_TOPIC_KEYWORDS)
    
    # Check for health-related keywords
    has_health_keywords = any(keyword in message_lower for keyword in HEALTH_RELATED_KEYWORDS)
    
    # If it has off-topic keywords and no health keywords, it's likely off-topic
    if has_off_topic and not has_health_keywords:
        return True
    
    return False

def validate_topic_with_ai(message: str) -> bool:
    """Use AI to validate if the question is related to reproductive health."""
    try:
        validation_response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": TOPIC_VALIDATION_PROMPT.format(message=message)}
            ],
            model=MODEL_NAME,
            temperature=0.3,  # Lower temperature for more consistent classification
            max_tokens=10
        )
        
        classification = validation_response.choices[0].message.content.strip().upper()
        return "RELEVANT" in classification
        
    except Exception:
        # If AI validation fails, be permissive and allow the question
        return True

def get_safety_response(message: str) -> Optional[str]:
    if check_emergency(message):
        return """🚨 **URGENT: Your message indicates a potentially serious medical situation.**

Please seek immediate medical attention:
- Call emergency services (911 in US, 112 in EU, or your local emergency number)
- Go to the nearest emergency room
- Contact your doctor immediately

Your health and safety are the top priority. Medical professionals can provide the urgent care you need."""

    if check_unsafe(message):
        return """I cannot provide information on this topic as it could be harmful to your health and safety.

If you're experiencing a crisis or having thoughts of self-harm:
- **Crisis Hotline:** 988 (Suicide & Crisis Lifeline - US)
- **International:** https://findahelpline.com

For reproductive health concerns, please speak with:
- A licensed healthcare provider
- Planned Parenthood or similar clinics
- A trusted counselor or therapist

Your wellbeing matters, and there are professionals ready to help you safely."""

    return None

def get_ai_response(message: str) -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            model=MODEL_NAME,
            temperature=0.7,
            max_tokens=500
        )
        
        ai_response = chat_completion.choices[0].message.content
        
        # Additional safety check
        if any(word in ai_response.lower() for word in ["i diagnose", "you have", "you need to take"]):
            ai_response += "\n\n⚠️ Remember: This is educational information only, not a diagnosis or prescription. Always consult a healthcare provider for personalized medical advice."
        
        return ai_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

# ============================================================================
# MENSTRUAL CYCLE PREDICTION SECTION
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

# Data preprocessing functions
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
    # Validate framework availability and auto-fallback
    if framework == 'tensorflow' and not TENSORFLOW_AVAILABLE:
        if PYTORCH_AVAILABLE:
            framework = 'pytorch'  # Auto-fallback to PyTorch
        else:
            raise HTTPException(
                status_code=500, 
                detail="TensorFlow is not available. PyTorch is the only supported framework on this server."
            )
    
    if framework == 'pytorch' and not PYTORCH_AVAILABLE:
        raise HTTPException(
            status_code=500, 
            detail="PyTorch is not installed on the server"
        )
    
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
        "service": "Combined Reproductive Health API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "chatbot": "Available at /chat",
            "cycle_prediction": "Available at /predict"
        },
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    frameworks = []
    if PYTORCH_AVAILABLE:
        frameworks.append("pytorch")
    if TENSORFLOW_AVAILABLE:
        frameworks.append("tensorflow")
    
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    
    return {
        "status": "healthy",
        "chatbot": {
            "status": "operational" if groq_configured else "not configured",
            "groq_configured": groq_configured,
            "model": MODEL_NAME
        },
        "cycle_predictor": {
            "status": "operational" if frameworks else "no ML frameworks available",
            "available_frameworks": frameworks
        },
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# CHATBOT ENDPOINTS
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the reproductive health education assistant.
    
    - **message**: Your question or message to the chatbot
    
    Returns educational information about reproductive health topics.
    """
    if not request.message or len(request.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if len(request.message) > 1000:
        raise HTTPException(status_code=400, detail="Message too long (max 1000 characters)")
    
    # Check for safety triggers first
    safety_response = get_safety_response(request.message)
    
    if safety_response:
        return ChatResponse(
            response=safety_response,
            safety_triggered=True
        )
    
    # Validate topic relevance - Two-layer approach
    # Layer 1: Quick keyword-based check
    if is_obviously_off_topic(request.message):
        return ChatResponse(
            response="""I'm a specialized reproductive health education assistant. I can only answer questions related to:

• Menstrual cycles and periods
• Pregnancy and fertility
• Reproductive health and anatomy
• Hormones and women's health
• Gynecological conditions (PCOS, endometriosis, etc.)

Your question appears to be about a different topic. Please ask me about reproductive health, and I'll be happy to help! 😊""",
            safety_triggered=False
        )
    
    # Layer 2: AI-powered validation for ambiguous cases
    if not validate_topic_with_ai(request.message):
        return ChatResponse(
            response="""I'm a specialized reproductive health education assistant. I can only answer questions related to:

• Menstrual cycles and periods
• Pregnancy and fertility
• Reproductive health and anatomy
• Hormones and women's health
• Gynecological conditions (PCOS, endometriosis, etc.)

Your question doesn't seem to be related to reproductive health. If you have questions about periods, pregnancy, fertility, or women's health, I'm here to help! 😊""",
            safety_triggered=False
        )
    
    # Get AI response for valid health-related questions
    ai_response = get_ai_response(request.message)
    
    return ChatResponse(
        response=ai_response,
        safety_triggered=False
    )

# ============================================================================
# CYCLE PREDICTION ENDPOINTS
# ============================================================================

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
    """List available ML frameworks for cycle prediction."""
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
    
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    print("=" * 70)
    print("🚀 COMBINED REPRODUCTIVE HEALTH API")
    print("=" * 70)
    print(f"📍 Server: http://localhost:{port}")
    print(f"📚 API Documentation: http://localhost:{port}/docs")
    print("=" * 70)
    print("CHATBOT STATUS:")
    print(f"  🤖 Model: {MODEL_NAME}")
    print(f"  🔑 Groq API Key: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing'}")
    print("=" * 70)
    print("CYCLE PREDICTOR STATUS:")
    print(f"  🔬 PyTorch: {'✅ Available' if PYTORCH_AVAILABLE else '❌ Not installed'}")
    print(f"  🔬 TensorFlow: {'✅ Available' if TENSORFLOW_AVAILABLE else '❌ Not installed'}")
    print("=" * 70)
    print("ENDPOINTS:")
    print("  💬 Chatbot: POST /chat")
    print("  📊 Cycle Prediction: POST /predict")
    print("  ❤️  Health Check: GET /health")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=port)
