# NaturAlert

**Stay ahead of nature's fury.** NaturAlert is an intelligent early warning system that combines machine learning, real-time weather data, and AI-powered insights to predict natural disasters before they strike. Whether you're tracking a hurricane, monitoring flood risks, or preparing for extreme weather, NaturAlert gives you the information you need to stay safe.

## Why NaturAlert?

Natural disasters claim thousands of lives and cause billions in damage every year. The difference between safety and catastrophe often comes down to minutes of warning. NaturAlert bridges that gap by analyzing weather patterns, historical disaster data, and current conditions to deliver accurate, actionable risk assessments in real-time.

## Screenshots

### Main Dashboard - Risk Assessment
![Main Dashboard](images/Screenshot%202025-11-23%20at%2016.19.08.png)

The main dashboard provides instant risk assessment for any location worldwide. Enter a city name or coordinates, and our machine learning model analyzes current weather conditions against historical disaster patterns. The interface displays a comprehensive risk score (0-100), real-time weather metrics including temperature, precipitation, wind speed, humidity, and pressure. Color-coded risk indicators make it easy to understand threat levels at a glance.

### Interactive 3D Globe Visualization
![3D Globe](images/Screenshot%202025-11-23%20at%2016.30.08.png)

Explore global weather patterns on our stunning 3D interactive globe. Rotate, zoom, and click on any region to see real-time weather conditions and disaster risk levels. The globe visualization helps you understand weather systems, track storms, and identify high-risk zones across the planet. Perfect for monitoring multiple locations or understanding regional weather patterns.

### AI Chat Assistant
![AI Chat](images/Screenshot%202025-11-23%20at%2016.30.25.png)

Get personalized guidance from our Google Gemini-powered AI assistant. Ask questions about weather risks, disaster preparedness, or specific locations. The AI provides intelligent, context-aware responses that help you understand complex weather phenomena and make informed decisions about safety. Whether you need evacuation advice or want to learn about historical disaster patterns, the AI assistant is here to help.

### Historical Trends Analysis
![Trends Analysis](images/Screenshot%202025-11-23%20at%2016.30.38.png)

Track how disaster risk evolves over time with our historical trends feature. View risk score changes across days, weeks, or months to identify patterns and seasonal variations. This feature helps you understand long-term climate trends, plan travel during safer periods, and make data-driven decisions about location-based activities.

### Emergency SOS Alert System
![SOS System](images/Screenshot%202025-11-23%20at%2016.31.42.png)

When disaster strikes, every second counts. Our SOS alert system lets you broadcast emergency alerts with your exact location. View nearby alerts from others in your area, coordinate with emergency responders, and help those in need. The system uses real-time geolocation to ensure accurate positioning and quick response times.

### Location Search & Analysis
![Location Search](images/Screenshot%202025-11-23%20at%2016.31.58.png)

Search any location on Earth with our powerful location search feature. Type city names, addresses, or coordinates to get instant risk assessments. The system provides detailed weather data, historical disaster context, and AI-generated safety recommendations for your searched location. Save favorite locations for quick access and monitor multiple areas simultaneously.

## What Makes It Special

**Intelligent Risk Assessment**  
Our machine learning models analyze multiple weather parameters against historical disaster patterns to calculate precise risk scores. You don't just get weather data—you get context, predictions, and actionable insights.

**Global Coverage, Local Precision**  
Search any location on Earth and get instant risk analysis. From major cities to remote villages, NaturAlert provides comprehensive disaster risk assessment wherever you need it.

**Interactive 3D Visualization**  
Explore weather patterns and risk zones on a stunning 3D globe. Watch storms develop, track weather systems, and understand global climate patterns at a glance.

**AI-Powered Guidance**  
Chat with our Google Gemini-powered assistant to understand complex weather phenomena, get personalized safety recommendations, and learn about disaster preparedness.

**Emergency SOS System**  
When disaster strikes, every second counts. Our SOS alert system lets you broadcast your location and status to emergency responders and loved ones instantly.

**Historical Context**  
Learn from the past to prepare for the future. Compare current conditions with historical disaster data to understand patterns and make informed decisions.

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

## How to Use NaturAlert

**Check Disaster Risk for Any Location**  
Type any city, address, or coordinates into the search box. Click "Get Risk Assessment" and within seconds you'll see a comprehensive risk score, current weather conditions, and an AI-generated explanation of potential threats. The system analyzes temperature, precipitation, wind speed, humidity, and pressure against historical disaster patterns to give you accurate predictions.

**Explore the Interactive Globe**  
Spin the 3D globe to explore weather patterns worldwide. Click on any region to see real-time conditions and risk levels. Watch weather systems move, identify high-risk zones, and understand global climate patterns visually.

**Get AI-Powered Insights**  
Open the AI Chat and ask anything: "What should I do if a hurricane is approaching?" or "Is it safe to travel to Miami next week?" Our Gemini-powered assistant provides personalized advice, explains complex weather phenomena, and helps you prepare for potential disasters.

**Use the SOS System in Emergencies**  
When disaster strikes, navigate to the SOS page and enable location access. Create an emergency alert that broadcasts your location and status. View nearby alerts to help others or coordinate with emergency responders.

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
├── .kiro/specs/          # Technical specifications and documentation
└── requirements.txt      # Python package dependencies
```

**Contributing to NaturAlert**

We welcome contributions that make disaster prediction more accurate and accessible:

1. Fork the repository and create a feature branch
2. Write clean, documented code that follows existing patterns
3. Test your changes thoroughly—lives may depend on this system
4. Submit a pull request with a clear description of your improvements

Whether you're fixing bugs, adding features, or improving documentation, your contributions help make the world safer.

## License

NaturAlert is open source software licensed under the MIT License. See the LICENSE file for complete terms. You're free to use, modify, and distribute this software, even for commercial purposes.

## Need Help?

Encountered a bug? Have a feature request? Want to discuss disaster prediction algorithms? Open an issue on GitHub and we'll respond as quickly as possible. For urgent matters related to emergency situations, please contact local authorities first.

## Acknowledgments

NaturAlert stands on the shoulders of giants. We're grateful to:

- Open-Meteo and OpenWeather for providing reliable, accessible weather data
- Google Gemini for powering our AI assistant with cutting-edge language models
- The Globe.gl team for making beautiful 3D visualizations possible
- The open source community for the incredible tools and libraries that make this project possible

## The Mission

Every year, natural disasters affect millions of people worldwide. Many of these tragedies could be prevented or mitigated with better early warning systems. NaturAlert exists to democratize access to sophisticated disaster prediction technology, making it available to everyone, everywhere, for free.

When you use NaturAlert, you're not just checking the weather—you're part of a global effort to save lives through technology.

---

**NaturAlert** - Because everyone deserves a warning.