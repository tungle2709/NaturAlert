# Frontend-Backend Integration Guide

This guide explains how to run the Disaster Early Warning System with both the backend API and frontend application.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│                  http://localhost:3000                   │
│                                                           │
│  - Dashboard with risk visualization                     │
│  - AI Chat interface (Gemini)                           │
│  - Weather data display                                  │
│  - Alert subscription                                    │
│  - SOS emergency system                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ REST API Calls
                     │
┌────────────────────▼────────────────────────────────────┐
│              Backend API (FastAPI)                       │
│                http://localhost:5000                     │
│                                                           │
│  - Risk assessment endpoints                             │
│  - Gemini AI integration                                 │
│  - Historical data queries                               │
│  - Map visualization data                                │
└────────────────────┬────────────────────────────────────┘
                     │
                     │
┌────────────────────▼────────────────────────────────────┐
│              Services & Data Layer                       │
│                                                           │
│  - Prediction Engine (ML models)                         │
│  - Gemini Service (AI explanations)                      │
│  - SQLite Database (disaster_data.db)                    │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### Backend Requirements
- Python 3.11 or 3.12
- Trained ML models in `models/` directory
- SQLite database with weather data
- Gemini API key (optional, for AI features)

### Frontend Requirements
- Node.js 18+ and npm
- Modern web browser

## Setup Instructions

### 1. Backend Setup

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### Configure Environment Variables
Create a `.env` file in the project root:
```bash
# Gemini API Key (optional - AI features will be disabled without it)
GEMINI_API_KEY=your_gemini_api_key_here

# Database Path (optional - defaults to disaster_data.db)
DATABASE_PATH=disaster_data.db
```

#### Verify Models and Database
Ensure these files exist:
- `models/disaster_prediction_model.pkl`
- `models/disaster_type_model.pkl`
- `disaster_data.db` (with weather_historical table populated)

### 2. Frontend Setup

#### Install Node Dependencies
```bash
cd frontend
npm install
```

#### Update App.jsx (Choose Integration Mode)

**Option A: Use the integrated version (recommended)**
```bash
# Backup original
mv src/App.jsx src/App.jsx.original

# Use integrated version
cp src/App-integrated.jsx src/App.jsx
```

**Option B: Keep original and manually integrate**
- Import the API service: `import * as api from './services/api';`
- Replace mock data calls with API calls

## Running the Application

### Terminal 1: Start Backend API

```bash
# From project root
python backend/app.py
```

Expected output:
```
============================================================
🌊 Disaster Early Warning System API
============================================================
📍 Server: http://localhost:5000
📚 API Docs: http://localhost:5000/docs
🔍 Health: http://localhost:5000/health
============================================================

✓ Loaded disaster prediction models from models
✓ Gemini AI service initialized
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:5000
```

### Terminal 2: Start Frontend Development Server

```bash
# From frontend directory
cd frontend
npm run dev
```

Expected output:
```
  VITE v5.0.8  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Access the Application

1. **Frontend UI**: http://localhost:5173 (or http://localhost:3000)
2. **Backend API**: http://localhost:5000
3. **API Documentation**: http://localhost:5000/docs (Interactive Swagger UI)

## Testing the Integration

### 1. Test Backend API Directly

Visit http://localhost:5000/docs to access the interactive API documentation.

Try these endpoints:
- `GET /health` - Check API health
- `GET /api/v1/risk/current?location_id=default` - Get risk assessment
- `GET /api/v1/risk/trends?location_id=default` - Get trend data

### 2. Test Frontend Integration

1. Open http://localhost:5173 in your browser
2. Enter a location ID (e.g., "default")
3. Click "Get Risk Assessment"
4. Verify that:
   - Risk score is displayed
   - Weather data appears
   - AI explanation shows (if Gemini is configured)

### 3. Test AI Chat (if Gemini is configured)

1. Click the "💬 AI Chat" tab
2. Ask a question like "What is the current risk level?"
3. Verify AI responds with contextual information

## API Endpoints Reference

### Risk Assessment
- `GET /api/v1/risk/current?location_id={id}` - Current risk
- `GET /api/v1/risk/trends?location_id={id}&days={n}` - Trend data
- `GET /api/v1/predictions/hourly?location_id={id}&hours={n}` - Hourly predictions

### Gemini AI
- `POST /api/v1/gemini/explain` - Get AI explanation
- `POST /api/v1/gemini/chat` - Chat with AI assistant

### Historical Data
- `GET /api/v1/history/disasters` - Query historical disasters
- `GET /api/v1/history/disasters/{id}` - Get disaster details

### Map & Visualization
- `GET /api/v1/map/heatmap` - Get heatmap data
- `GET /api/v1/map/markers?marker_type={type}` - Get map markers

## Troubleshooting

### Backend Issues

**Problem**: `FileNotFoundError: Binary model not found`
```bash
# Solution: Ensure models are trained
python notebooks/03_model_training.ipynb
# Or check that .pkl files exist in models/
```

**Problem**: `ValueError: Insufficient weather data`
```bash
# Solution: Populate database with weather data
python notebooks/02_data_preprocessing.ipynb
```

**Problem**: `Gemini service initialization failed`
```bash
# Solution: Check GEMINI_API_KEY in .env file
# Or disable Gemini by setting use_gemini=False in backend/app.py
```

### Frontend Issues

**Problem**: `Failed to fetch` or CORS errors
```bash
# Solution: Ensure backend is running on port 5000
# Check CORS configuration in backend/app.py includes your frontend URL
```

**Problem**: Frontend shows old mock data
```bash
# Solution: Clear browser cache or use incognito mode
# Verify you're using App-integrated.jsx
```

### Network Issues

**Problem**: Cannot connect to backend
```bash
# Check if backend is running
curl http://localhost:5000/health

# Check if port 5000 is available
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows
```

## Development Workflow

### Making Backend Changes

1. Edit files in `backend/`
2. Restart backend server (Ctrl+C, then `python backend/app.py`)
3. Test changes via API docs or frontend

### Making Frontend Changes

1. Edit files in `frontend/src/`
2. Vite will auto-reload (no restart needed)
3. Check browser console for errors

### Adding New API Endpoints

1. Add endpoint to `backend/app.py`
2. Add corresponding function to `frontend/src/services/api.js`
3. Use the new function in your React components

## Production Deployment Notes

For production deployment:

1. **Backend**:
   - Use production WSGI server (gunicorn)
   - Set up proper environment variables
   - Use PostgreSQL instead of SQLite
   - Enable HTTPS

2. **Frontend**:
   - Build production bundle: `npm run build`
   - Serve static files via nginx or CDN
   - Update API_BASE_URL to production backend URL

3. **Security**:
   - Enable authentication/authorization
   - Set up rate limiting
   - Use environment-specific CORS settings
   - Secure API keys and secrets

## Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **React Documentation**: https://react.dev/
- **Gemini API**: https://ai.google.dev/docs
- **Project README**: See README.md for overall project documentation

## Support

If you encounter issues:
1. Check the console logs (both backend and frontend)
2. Verify all prerequisites are installed
3. Ensure database and models are properly set up
4. Check the troubleshooting section above
