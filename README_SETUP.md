# Disaster Early Warning System - Setup Guide

## Prerequisites

- **Python 3.11 or 3.12** (Python 3.14 has compatibility issues with some dependencies)
- pip (Python package manager)
- Virtual environment tool (venv)

## Installation Steps

### 1. Check Python Version

```bash
python3 --version
```

If you have Python 3.14, you'll need to install Python 3.11 or 3.12:
- macOS: `brew install python@3.12`
- Ubuntu/Debian: `sudo apt install python3.12`
- Windows: Download from python.org

### 2. Create Virtual Environment

```bash
# Using Python 3.12
python3.12 -m venv venv

# Or using Python 3.11
python3.11 -m venv venv
```

### 3. Activate Virtual Environment

```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
- `GEMINI_API_KEY`: Your Google Gemini API key

### 6. Initialize Database

```bash
python backend/database/init_db.py
```

## Project Structure

```
.
├── backend/           # Backend API code
│   ├── api/          # API endpoints
│   ├── services/     # Business logic
│   ├── models/       # Data models
│   ├── utils/        # Utility functions
│   └── database/     # Database setup
├── frontend/         # React frontend (to be created)
├── notebooks/        # Jupyter notebooks for data analysis
├── models/           # Trained ML models (.pkl files)
├── data/             # Processed data
└── dataset/          # Raw CSV datasets

```

## Next Steps

1. Run data preprocessing notebooks to prepare training data
2. Train ML models
3. Start the backend API server
4. Set up the React frontend

## Troubleshooting

### Python Version Issues

If you encounter compatibility errors, ensure you're using Python 3.11 or 3.12:

```bash
python --version
```

### Dependency Installation Failures

Try installing dependencies one at a time to identify the problematic package:

```bash
pip install fastapi
pip install pandas
# etc.
```

### Database Issues

If the database initialization fails, check:
- Write permissions in the project directory
- SQLite is available (usually built into Python)
