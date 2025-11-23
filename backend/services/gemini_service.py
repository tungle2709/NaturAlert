"""
Gemini AI Integration Service

This module provides integration with Google's Gemini API for generating
natural language explanations, chat responses, and AI-powered insights
for the Disaster Early Warning System.
"""

import os
from typing import Dict, Optional, List
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai package not installed. Gemini features will be disabled.")


class GeminiService:
    """
    Service class for interacting with Google's Gemini API.
    
    Provides methods for:
    - Generating risk explanations
    - Chat functionality
    - Feature importance explanations
    - Alert message generation
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini service.
        
        Args:
            api_key: Optional API key. If not provided, will load from GEMINI_API_KEY env var.
        
        Raises:
            ValueError: If API key is not provided and not found in environment
            RuntimeError: If google-generativeai package is not installed
        """
        if not GEMINI_AVAILABLE:
            raise RuntimeError(
                "google-generativeai package is not installed. "
                "Install it with: pip install google-generativeai"
            )
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in .env file or pass as parameter."
            )
        
        # Configure Gemini API
        try:
            genai.configure(api_key=self.api_key)
            # Try gemini-2.5-flash first, fallback to gemini-1.5-flash
            try:
                self.model = genai.GenerativeModel('gemini-2.5-flash')
            except:
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            self._initialized = True
        except Exception as e:
            self._initialized = False
            raise RuntimeError(f"Failed to initialize Gemini API: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Gemini service is properly initialized."""
        return self._initialized
    
    def _safe_generate(self, prompt: str, fallback_message: str) -> str:
        """
        Safely generate content with error handling.
        
        Args:
            prompt: The prompt to send to Gemini
            fallback_message: Message to return if generation fails
            
        Returns:
            Generated text or fallback message
        """
        if not self._initialized:
            return fallback_message
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return fallback_message
    
    def generate_explanation(
        self, 
        weather_data: pd.DataFrame, 
        prediction: Dict
    ) -> str:
        """
        Generate natural language explanation for a disaster prediction.
        
        Args:
            weather_data: DataFrame containing recent weather observations
            prediction: Dictionary containing ML prediction results with keys:
                - risk_score: float (0-100)
                - disaster_type: str
                - confidence: float (0-100)
        
        Returns:
            Natural language explanation string
        """
        # Get latest weather snapshot
        if len(weather_data) == 0:
            return "Insufficient weather data for explanation."
        
        latest = weather_data.iloc[-1]
        
        # Build prompt
        prompt = f"""You are a disaster early warning system AI assistant.

Current weather conditions:
- Temperature: {latest.get('temperature', 'N/A'):.1f}°C
- Pressure: {latest.get('pressure', 'N/A'):.1f} hPa
- Humidity: {latest.get('humidity', 'N/A'):.1f}%
- Wind Speed: {latest.get('wind_speed', 'N/A'):.1f} mph
- 24h Rainfall: {latest.get('rainfall_24h', 0):.1f} mm

ML Prediction:
- Risk Score: {prediction['risk_score']:.1f}%
- Disaster Type: {prediction['disaster_type']}
- Confidence: {prediction['confidence']:.1f}%

Explain in 2-3 sentences why the risk is at this level, referencing specific weather patterns that match historical pre-disaster conditions. Be clear and actionable."""
        
        fallback = (
            f"The current risk level is {prediction['risk_score']:.0f}% for {prediction['disaster_type']}. "
            f"This assessment is based on current weather patterns including "
            f"pressure at {latest.get('pressure', 'N/A'):.0f} hPa and wind speed at {latest.get('wind_speed', 'N/A'):.0f} mph."
        )
        
        return self._safe_generate(prompt, fallback)
    
    def chat_response(
        self, 
        message: str, 
        context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a chat response with contextual awareness and conversation history.
        
        Args:
            message: User's question or message
            context: Optional context dictionary containing:
                - location: str
                - risk_score: float
                - temperature: float
                - pressure: float
                - disaster_type: str
                - etc.
            conversation_history: Optional list of previous messages in format:
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            AI-generated response string
        """
        context = context or {}
        conversation_history = conversation_history or []
        
        # Build context string
        context_str = "Current context:\n"
        if 'location' in context:
            context_str += f"- Location: {context['location']}\n"
        if 'risk_score' in context:
            context_str += f"- Current Risk: {context['risk_score']:.1f}%\n"
        if 'disaster_type' in context:
            context_str += f"- Disaster Type: {context['disaster_type']}\n"
        if 'temperature' in context and 'pressure' in context:
            context_str += f"- Recent Weather: Temperature {context['temperature']:.1f}°C, Pressure {context['pressure']:.1f} hPa\n"
        
        # Build conversation history string
        history_str = ""
        if conversation_history:
            history_str = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-5:]:  # Only include last 5 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_str += f"{role.capitalize()}: {content}\n"
        
        prompt = f"""You are a disaster early warning assistant helping users understand weather risks and disaster preparedness.

{context_str}{history_str}

User question: {message}

Provide a helpful, accurate response. Include relevant data and historical comparisons when appropriate. Keep responses concise (3-4 sentences max). Be reassuring but honest about risks."""
        
        fallback = (
            "I'm here to help with disaster preparedness and weather risk information. "
            "Could you please rephrase your question?"
        )
        
        return self._safe_generate(prompt, fallback)
    
    def generate_feature_importance_explanation(
        self, 
        feature_importance: Dict[str, float]
    ) -> str:
        """
        Generate explanations for ML feature importance.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance percentages
        
        Returns:
            Natural language explanation of feature importance
        """
        # Build features text
        features_text = '\n'.join([
            f"- {feature}: {importance:.1f}%"
            for feature, importance in sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        ])
        
        prompt = f"""Explain why these weather features are important for disaster prediction:

{features_text}

Provide a clear explanation for each feature in simple terms that a non-technical person can understand. Explain what each feature measures and why it matters for predicting disasters."""
        
        fallback = (
            "These features represent key weather patterns that historically precede disasters. "
            "Higher importance percentages indicate stronger predictive power."
        )
        
        return self._safe_generate(prompt, fallback)
    
    def generate_alert_message(self, prediction: Dict) -> str:
        """
        Generate a concise, actionable alert message.
        
        Args:
            prediction: Dictionary containing:
                - risk_score: float (0-100)
                - disaster_type: str
                - confidence: float (0-100)
        
        Returns:
            User-friendly alert message
        """
        prompt = f"""Generate a concise, urgent alert message for users about an impending disaster.

Risk Score: {prediction['risk_score']:.1f}%
Disaster Type: {prediction['disaster_type']}
Confidence: {prediction['confidence']:.1f}%

The message should:
1. Be 1-2 sentences
2. Be clear and actionable
3. Not cause panic but convey urgency
4. Suggest basic preparedness actions

Example: "High flood risk detected (85%). Secure valuables and monitor local emergency updates."
"""
        
        fallback = (
            f"⚠️ {prediction['disaster_type'].replace('_', ' ').title()} risk: "
            f"{prediction['risk_score']:.0f}%. Stay alert and monitor conditions."
        )
        
        return self._safe_generate(prompt, fallback)


# Singleton instance for easy import
_gemini_service_instance = None


def get_gemini_service() -> GeminiService:
    """
    Get or create the singleton Gemini service instance.
    
    Returns:
        GeminiService instance
    
    Raises:
        RuntimeError: If service cannot be initialized
    """
    global _gemini_service_instance
    
    if _gemini_service_instance is None:
        _gemini_service_instance = GeminiService()
    
    return _gemini_service_instance
