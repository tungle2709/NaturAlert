# 🌐 Ngrok Public URLs

## Active Tunnels

### 🔧 Backend API
```
https://c368b243f830.ngrok-free.app
```
- **Local Port**: 8000
- **Status**: ✅ Online
- **API Docs**: https://c368b243f830.ngrok-free.app/docs
- **Health Check**: https://c368b243f830.ngrok-free.app/health

### 🎨 Frontend Application
```
https://6641072add9a.ngrok-free.app
```
- **Local Port**: 5173
- **Status**: ✅ Online
- **Access**: Open in any browser worldwide

## 📊 Ngrok Web Interface
```
http://127.0.0.1:4040
```
Monitor all traffic, inspect requests, and replay them.

## 🧪 Test Your APIs

### Backend Health Check
```bash
curl "https://c368b243f830.ngrok-free.app/health"
```

### Disaster Risk Assessment
```bash
curl "https://c368b243f830.ngrok-free.app/api/v1/risk/current?location_id=43.65,-79.38"
```

### Frontend Access
Simply open in browser:
```
https://6641072add9a.ngrok-free.app
```

## 📱 Share Your App

You can now share these URLs with anyone:

**Frontend (Full App):**
```
https://6641072add9a.ngrok-free.app
```

**Backend API:**
```
https://c368b243f830.ngrok-free.app
```

## ⚙️ Configuration

The tunnels are configured in `/tmp/ngrok_config.yml`:
```yaml
version: "2"
authtoken: 2kdO3tpcFiSrhiAIcus8YZBiWjQ_6UGjpbVoF7peqzf9e31Cw
tunnels:
  backend:
    proto: http
    addr: 8000
  frontend:
    proto: http
    addr: 5173
```

## 🔄 Restart Tunnels

If you need to restart:
```bash
ngrok start --all --config /tmp/ngrok_config.yml
```

## ⚠️ Important Notes

1. **First Visit Warning**: Users will see an ngrok warning page. Click "Visit Site" to continue.
2. **URL Changes**: These URLs will change if you restart ngrok (free tier).
3. **Keep Running**: Keep ngrok running for the tunnels to work.
4. **Security**: Your apps are now public - anyone with the URLs can access them.

## 🛑 Stop Tunnels

To stop all tunnels, stop the ngrok process.

## 📈 Upgrade Options

Free tier limitations:
- ⚠️ Random URLs (change on restart)
- ⚠️ Warning page on first visit
- ⚠️ 1 agent session (but multiple tunnels via config)

Paid tier ($8/month):
- ✅ Custom subdomains (e.g., myapp.ngrok.app)
- ✅ No warning page
- ✅ Reserved domains
- ✅ More concurrent tunnels

## 🎯 Current Status

- ✅ Backend running on port 8000
- ✅ Frontend running on port 5173
- ✅ Both exposed via ngrok
- ✅ Frontend configured to use backend ngrok URL
- ✅ Ready to share worldwide!

---

**Last Updated**: 2025-11-23
**Ngrok Version**: 3.33.0
**Account**: giangma (Free Plan)
