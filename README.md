# NaturAlert

**Stay ahead of nature's fury.** NaturAlert is an intelligent early warning system that combines machine learning, real-time weather data, and AI-powered insights to predict natural disasters before they strike. Whether you're tracking a hurricane, monitoring flood risks, or preparing for extreme weather, NaturAlert gives you the information you need to stay safe.

## Why NaturAlert?

Natural disasters claim thousands of lives and cause billions in damage every year. The difference between safety and catastrophe often comes down to minutes of warning. NaturAlert bridges that gap by analyzing weather patterns, historical disaster data, and current conditions to deliver accurate, actionable risk assessments in real-time.

## Data Science for Social Good

NaturAlert exemplifies how data science and machine learning can be powerful forces for positive social impact. By leveraging advanced analytics, predictive modeling, and AI, we're addressing critical challenges that affect millions of people worldwide. Our work directly contributes to four United Nations Sustainable Development Goals:

### SDG 11: Sustainable Cities and Communities

**Making Urban Areas Safer and More Resilient**

Cities are home to over half the world's population and are particularly vulnerable to natural disasters. NaturAlert helps urban planners, emergency managers, and residents make data-driven decisions about disaster preparedness and response.

- **Urban Risk Mapping**: Our ML models identify high-risk urban zones by analyzing historical disaster patterns, population density, and infrastructure vulnerability
- **Early Warning Systems**: Real-time alerts give city residents crucial minutes or hours to evacuate, secure property, or seek shelter
- **Infrastructure Planning**: Historical trend analysis helps city planners identify areas requiring improved drainage, flood barriers, or emergency shelters
- **Community Resilience**: By democratizing access to disaster prediction technology, we empower communities to build their own resilience strategies

Data science enables us to process vast amounts of weather data, satellite imagery, and historical records to create actionable insights that save lives and protect urban infrastructure.

### SDG 13: Climate Action

**Understanding and Responding to Climate Change**

Climate change is intensifying natural disasters worldwide. NaturAlert uses data science to help communities understand, adapt to, and mitigate climate-related risks.

- **Pattern Recognition**: Machine learning algorithms detect long-term climate trends and changing disaster patterns that might not be visible to human analysts
- **Predictive Analytics**: Our models forecast how climate change will affect disaster frequency and intensity in specific regions
- **Data-Driven Advocacy**: Historical trend data provides evidence for climate action policies and resource allocation decisions
- **Carbon-Conscious Design**: Our cloud-based architecture optimizes computational resources to minimize environmental impact while maximizing prediction accuracy

By analyzing decades of climate and disaster data, we help communities prepare for a changing climate and make informed decisions about adaptation strategies.

### SDG 3: Good Health and Well-Being

**Protecting Lives Through Predictive Healthcare**

Natural disasters have profound impacts on public health, from immediate injuries to long-term mental health effects and disease outbreaks. NaturAlert's predictive capabilities help healthcare systems prepare and respond effectively.

- **Preventive Health**: Early warnings allow people with medical conditions to secure medications, evacuate safely, and avoid health emergencies
- **Healthcare System Preparedness**: Hospitals and clinics can pre-position resources, staff, and supplies based on predicted disaster impacts
- **Mental Health Support**: Reducing uncertainty through accurate predictions helps minimize anxiety and trauma associated with disasters
- **Disease Prevention**: Predicting floods and extreme weather helps public health officials prepare for waterborne diseases and other health risks

Our AI assistant provides personalized safety recommendations for vulnerable populations, including elderly individuals, people with disabilities, and those with chronic health conditions.

### SDG 15: Life on Land

**Protecting Ecosystems and Biodiversity**

Natural disasters don't just affect humans—they devastate ecosystems, wildlife habitats, and biodiversity. NaturAlert's predictive capabilities support conservation efforts and ecosystem protection.

- **Wildlife Protection**: Early warnings enable wildlife managers to implement emergency protocols, relocate endangered species, or secure protected areas
- **Ecosystem Monitoring**: Our global coverage helps track how extreme weather events affect forests, wetlands, and other critical ecosystems
- **Agricultural Planning**: Farmers can protect crops, livestock, and soil health by preparing for predicted extreme weather events
- **Reforestation Support**: Historical disaster data helps identify resilient planting locations and optimal times for reforestation projects

By analyzing weather patterns and disaster risks, we help conservationists and land managers make data-driven decisions that protect biodiversity and ecosystem services.

## The Power of Open Data Science

NaturAlert is built on the principle that life-saving technology should be accessible to everyone. We use open-source tools, public datasets, and transparent methodologies to ensure our work can be replicated, improved, and adapted by communities worldwide.

**Our Data Science Approach:**

- **Ethical AI**: Our models are trained on diverse, representative datasets to avoid bias and ensure equitable protection for all communities
- **Transparency**: We document our methodologies, share our code, and explain our predictions so users can trust and understand our system
- **Continuous Learning**: Our models improve over time by learning from new data, user feedback, and emerging climate patterns
- **Collaborative Science**: We welcome contributions from data scientists, climate researchers, and domain experts to improve prediction accuracy

**Impact Through Technology:**

- Analyzing millions of weather data points daily to identify emerging threats
- Processing historical disaster records spanning decades to understand patterns
- Leveraging AI to make complex climate science accessible to non-experts
- Providing free, global coverage so no community is left behind

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