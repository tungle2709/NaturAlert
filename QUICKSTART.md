# Quick Start Guide 🚀

Get the Disaster Early Warning System running in 3 minutes!

## Prerequisites Check

```bash
# Check Python (need 3.11+)
python --version

# Check Node.js (need 18+)
node --version

# Check if models exist
ls models/*.pkl

# Check if database exists
ls disaster_data.db
```

## Step 1: Start Backend (Terminal 1)

```bash
# From project root
python backend/app.py
```

✅ You should see:
```
============================================================
🌊 Disaster Early Warning System API
============================================================
📍 Server: http://localhost:5000
```

## Step 2: Start Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

✅ You should see:
```
➜  Local:   http://localhost:5173/
```

## Step 3: Open Browser

Navigate to: **http://localhost:5173**

## Step 4: Test It!

1. **Enter location**: Type `default` in the search box
2. **Click**: "Get Risk Assessment"
3. **See results**:
   - Risk score (e.g., 75.5%)
   - Weather data (temperature, pressure, etc.)
   - AI explanation (if Gemini configured)

## Try AI Chat

1. Click **💬 AI Chat** tab
2. Ask: "What is the current risk level?"
3. Get AI-powered response!

## Quick Test Commands

```bash
# Test backend health
curl http://localhost:5000/health

# Test risk endpoint
curl http://localhost:5000/api/v1/risk/current?location_id=default

# Test heatmap endpoint
curl http://localhost:5000/api/v1/map/heatmap
```

## Troubleshooting

### Backend won't start?
```bash
pip install -r requirements.txt
```

### Frontend won't start?
```bash
cd frontend
npm install
```

### No data showing?
Check that `disaster_data.db` has weather data:
```bash
sqlite3 disaster_data.db "SELECT COUNT(*) FROM weather_historical;"
```

If empty, run data preprocessing:
```bash
# Run the Jupyter notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_data_preprocessing.ipynb
# 3. notebooks/03_model_training.ipynb
```

## Optional: Enable AI Features

Create `.env` file:
```bash
GEMINI_API_KEY=your_api_key_here
```

Restart backend to enable AI explanations and chat.

## Success! 🎉

You should now see:
- ✅ Risk scores from ML models
- ✅ Real weather data
- ✅ Interactive dashboard
- ✅ AI chat (if Gemini configured)

## Next Steps

- Explore different locations
- Try the AI chat assistant
- Check out the API docs: http://localhost:5000/docs
- Read INTEGRATION_COMPLETE.md for full details

## Need Help?

- Check console logs (browser F12 and terminal)
- Review INTEGRATION_GUIDE.md
- Visit API docs: http://localhost:5000/docs
