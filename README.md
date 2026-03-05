# NaturAlert

**Stay ahead of nature's fury.** NaturAlert is an intelligent early warning system that combines machine learning, real-time weather data, and AI-powered insights to predict natural disasters before they strike. Whether you're tracking a hurricane, monitoring flood risks, or preparing for extreme weather, NaturAlert gives you the information you need to stay safe.

## Why NaturAlert?

Natural disasters claim thousands of lives and cause billions in damage every year. The difference between safety and catastrophe often comes down to minutes of warning. NaturAlert bridges that gap by analyzing weather patterns, historical disaster data, and current conditions to deliver accurate, actionable risk assessments in real-time.

## The Power of Open Data Science

NaturAlert is built on the principle that life-saving technology should be accessible to everyone. We use open-source tools, public datasets, and transparent methodologies to ensure our work can be replicated, improved, and adapted by communities worldwide.

## Built With Modern Technology

**Backend Infrastructure**  
Python 3.9+ powers our FastAPI backend, delivering lightning-fast API responses. Machine learning models built with scikit-learn analyze weather patterns, while Google Gemini AI provides intelligent conversational insights. Data persistence handled by SQLite in development and PostgreSQL for production deployments.

**Frontend Experience**  
React 18 with Vite ensures a blazing-fast, responsive user interface. Globe.gl brings stunning 3D visualizations to life, while Tailwind CSS provides a clean, modern design. Firebase Authentication keeps your data secure, and WebSocket integration delivers real-time updates as conditions change.

**Data Sources**  
Real-time weather data from Open-Meteo and OpenWeather APIs ensures accuracy and reliability. Historical disaster data helps our models learn from past events to predict future risks.

## Get Started in Minutes

**What You'll Need**  
Python 3.9 or higher, Node.js 18 or higher, and Git. That's it.

**Installation**

Clone and set up the project:

```bash
# Get the code
git clone https://github.com/tungle2709/NaturAlert.git
cd NaturAlert

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install --legacy-peer-deps
cd ..

# Optional: Add your Gemini API key for AI features
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**Launch the Application**

Start the backend server:
```bash
python3 backend/app.py
```
The API will be available at http://localhost:8000

In a new terminal, start the frontend:
```bash
cd frontend
npm run dev
```
The app will open at http://localhost:5173

Open your browser and start exploring. No complex configuration required.


## API Reference

NaturAlert provides a RESTful API for developers who want to integrate disaster prediction into their own applications:

- `GET /health` - Check system status and uptime
- `GET /api/v1/risk/current` - Get real-time risk assessment for any location
- `GET /api/v1/risk/trends` - Access historical trend analysis and patterns
- `POST /api/v1/gemini/chat` - Interact with the AI assistant programmatically
- `GET /docs` - Explore interactive API documentation with live examples

Visit http://localhost:8000/docs when running locally to test endpoints in your browser.

## For Developers

**Project Architecture**

The codebase is organized for clarity and maintainability:

```
├── backend/                 # FastAPI server and business logic
│   ├── services/           # ML models, AI integration, weather APIs
│   ├── database/           # Data persistence and schema
│   └── app.py             # Application entry point
├── frontend/               # React application
│   ├── src/               # Components, services, and utilities
│   └── public/            # Static assets and resources
├── models/                # Trained machine learning models
└── requirements.txt      # Python package dependencies
```

**Contributing to NaturAlert**

We welcome contributions that make disaster prediction more accurate and accessible:

1. Fork the repository and create a feature branch
2. Write clean, documented code that follows existing patterns
3. Test your changes thoroughly—lives may depend on this system
4. Submit a pull request with a clear description of your improvements

Whether you're fixing bugs, adding features, or improving documentation, your contributions help make the world safer.

---

**NaturAlert** - Because everyone deserves a warning.
