# Disaster Early Warning System

A comprehensive disaster prediction and early warning system that uses machine learning and AI to assess weather-related disaster risks in real-time.

## Features

- **Real-time Risk Assessment**: ML-powered disaster prediction using weather data
- **Interactive 3D Globe**: Visualize global weather patterns and risk zones
- **Location Search**: Search and analyze any location worldwide
- **SOS Alert System**: Emergency alert system with geolocation
- **AI Chat Assistant**: Powered by Google Gemini for weather insights
- **Historical Analysis**: Compare current conditions with historical disaster patterns
- **Multi-disaster Detection**: Floods, storms, hurricanes, extreme weather

## Technology Stack

### Backend
- **Python 3.9+** with FastAPI
- **Machine Learning**: scikit-learn, joblib
- **AI Integration**: Google Gemini API
- **Database**: SQLite (development), PostgreSQL (production)
- **Weather APIs**: Open-Meteo, OpenWeather

### Frontend
- **React 18** with Vite
- **3D Visualization**: Globe.gl
- **Styling**: Tailwind CSS
- **Authentication**: Firebase Auth
- **Real-time Updates**: WebSocket integration

## Quick Start

### Prerequisites
- Python 3.9 or higher
- Node.js 18 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tungle2709/Mien-Trung-System.git
   cd Mien-Trung-System
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install --legacy-peer-deps
   cd ..
   ```

4. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY for AI features
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   python3 backend/app.py
   ```
   Backend will run on: http://localhost:8000

2. **Start the frontend server** (in a new terminal)
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend will run on: http://localhost:5173

3. **Open your browser**
   Navigate to: http://localhost:5173

## Usage

### Basic Risk Assessment
1. Enter a location in the search box (e.g., "Toronto", "New York")
2. Click "Get Risk Assessment"
3. View the risk score, weather data, and AI explanation

### SOS Alerts
1. Go to the SOS page
2. Enable location access
3. Create or view emergency alerts in your area

### AI Chat
1. Click the "AI Chat" tab
2. Ask questions about weather risks and disaster preparedness
3. Get AI-powered insights and recommendations

## API Endpoints

- `GET /health` - Health check
- `GET /api/v1/risk/current` - Current risk assessment
- `GET /api/v1/risk/trends` - Historical trend data
- `POST /api/v1/gemini/chat` - AI chat interface
- `GET /docs` - Interactive API documentation

## Development

### Project Structure
```
├── backend/                 # Python FastAPI backend
│   ├── services/           # Core services (ML, AI)
│   ├── database/           # Database schema and utilities
│   └── app.py             # Main application entry point
├── frontend/               # React frontend
│   ├── src/               # Source code
│   └── public/            # Static assets
├── models/                # Trained ML models
├── .kiro/specs/          # Project specifications
└── requirements.txt      # Python dependencies
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or support, please open an issue on GitHub or contact the development team.

## Acknowledgments

- Weather data provided by Open-Meteo and OpenWeather APIs
- AI capabilities powered by Google Gemini
- 3D globe visualization using Globe.gl
- Built with modern web technologies and best practices