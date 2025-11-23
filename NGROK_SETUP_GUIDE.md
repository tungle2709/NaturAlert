# Ngrok Setup Guide

## What is Ngrok?
Ngrok creates a secure tunnel from a public URL to your localhost, allowing you to expose your local backend API to the internet.

## Setup Steps

### 1. Sign Up for Ngrok (Free)
Visit: https://dashboard.ngrok.com/signup

- Sign up with your email or GitHub account
- It's completely free for basic usage

### 2. Get Your Authtoken
After signing up:
1. Go to: https://dashboard.ngrok.com/get-started/your-authtoken
2. Copy your authtoken (it looks like: `2abc123def456ghi789jkl0mnop1qrs_2stu3vwx4yz5ABC6DEF7GHI`)

### 3. Configure Ngrok
Run this command with YOUR authtoken:
```bash
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

Example:
```bash
ngrok config add-authtoken 2abc123def456ghi789jkl0mnop1qrs_2stu3vwx4yz5ABC6DEF7GHI
```

### 4. Start Ngrok Tunnel
Once configured, run:
```bash
ngrok http 8000
```

This will create a public URL that forwards to your localhost:8000 backend.

## What You'll Get

After starting ngrok, you'll see output like:
```
ngrok                                                              

Session Status                online
Account                       your-email@example.com
Version                       3.33.0
Region                        United States (us)
Latency                       -
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://abc123def456.ngrok-free.app -> http://localhost:8000

Connections                   ttl     opn     rt1     rt5     p50     p90
                              0       0       0.00    0.00    0.00    0.00
```

### Your Public URL
The **Forwarding** line shows your public URL:
- Example: `https://abc123def456.ngrok-free.app`
- This URL is accessible from anywhere on the internet
- It forwards all requests to your localhost:8000

## Using Your Public URL

### Test Your API
```bash
# Instead of localhost
curl "http://localhost:8000/api/v1/risk/current?location_id=43.65,-79.38"

# Use your ngrok URL
curl "https://abc123def456.ngrok-free.app/api/v1/risk/current?location_id=43.65,-79.38"
```

### Update Frontend
If you want your frontend to use the public URL, update `frontend/src/services/api.js`:

```javascript
// Change from:
const API_BASE_URL = 'http://localhost:8000';

// To your ngrok URL:
const API_BASE_URL = 'https://abc123def456.ngrok-free.app';
```

## Important Notes

### Free Tier Limitations
- ✅ HTTPS tunnel
- ✅ Random subdomain (e.g., abc123def456.ngrok-free.app)
- ✅ No time limit
- ⚠️ URL changes each time you restart ngrok
- ⚠️ Shows ngrok warning page on first visit (users can click "Visit Site")

### Paid Tier Benefits ($8/month)
- Custom subdomain (e.g., myapp.ngrok.app)
- No warning page
- More concurrent tunnels
- Reserved domains

### Security Considerations
- 🔒 Your backend is now accessible from the internet
- 🔒 Anyone with the URL can access your API
- 🔒 Consider adding authentication if exposing sensitive data
- 🔒 The URL is random and hard to guess (free tier)
- 🔒 You can stop ngrok anytime to close the tunnel

## Ngrok Web Interface

While ngrok is running, visit: http://127.0.0.1:4040

This shows:
- All incoming requests
- Request/response details
- Replay requests
- Inspect traffic

Very useful for debugging!

## Common Commands

```bash
# Start tunnel to port 8000
ngrok http 8000

# Start tunnel with custom subdomain (paid)
ngrok http 8000 --subdomain=myapp

# Start tunnel with specific region
ngrok http 8000 --region=us

# View help
ngrok help

# Check version
ngrok version

# View configuration
ngrok config check
```

## Troubleshooting

### "authentication failed"
- You need to add your authtoken first
- Run: `ngrok config add-authtoken YOUR_TOKEN`

### "tunnel not found"
- Make sure your backend is running on port 8000
- Check: `curl http://localhost:8000/health`

### "connection refused"
- Your backend might not be running
- Start it: `python3 backend/app.py`

### URL changes every restart
- This is normal for free tier
- Upgrade to paid for reserved domains
- Or use a dynamic DNS service

## Alternative: Cloudflare Tunnel (Free)

If you want a permanent URL without paying, consider Cloudflare Tunnel:
```bash
# Install
brew install cloudflare/cloudflare/cloudflared

# Login
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create my-tunnel

# Route tunnel
cloudflared tunnel route dns my-tunnel myapp.yourdomain.com

# Run tunnel
cloudflared tunnel run my-tunnel
```

Requires your own domain but is completely free.

## Quick Start (After Setup)

1. **Start Backend**:
   ```bash
   python3 backend/app.py
   ```

2. **Start Ngrok** (in another terminal):
   ```bash
   ngrok http 8000
   ```

3. **Copy Public URL** from ngrok output

4. **Test It**:
   ```bash
   curl "https://YOUR-NGROK-URL.ngrok-free.app/health"
   ```

5. **Share URL** with others or use in your frontend

## Next Steps

1. Sign up at: https://dashboard.ngrok.com/signup
2. Get authtoken: https://dashboard.ngrok.com/get-started/your-authtoken
3. Run: `ngrok config add-authtoken YOUR_TOKEN`
4. Run: `ngrok http 8000`
5. Copy your public URL and start using it!

---

**Need Help?**
- Ngrok Docs: https://ngrok.com/docs
- Ngrok Dashboard: https://dashboard.ngrok.com
- Support: https://ngrok.com/support
