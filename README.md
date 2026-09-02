# Adaptive-Anamoly-Residual-Early-Warning-System-for-Urban-traffic
### G. Pullaiah College of Engineering and Technology, Kurnool, A.P.

---

## How to Run

### Terminal 1 — Backend (Flask)
```
cd backend
pip install -r requirements.txt
python app.py
```
First run trains the N-TCN (~60 seconds). Wait for:
`Running inference on TEST set.`

### Terminal 2 — Frontend (React + Vite)
```
cd frontend
npm install
npm start
```
Opens at http://localhost:3000

---

## Requirements
- Python 3.8+
- Node.js 16+ (works on Node 24)
- Both terminals must stay open

---

## API Endpoints
| Endpoint | Description |
|----------|-------------|
| GET /api/status | Health check |
| GET /api/sensors | Sensor list |
| GET /api/traffic | One pipeline step (Eq.1–10) |
| GET /api/history/<id> | Sensor history |
| GET /api/metrics | System metrics |
| GET /api/alerts | Alert log |
| GET /api/evaluation | Lead time + F1 (Eq.16–17) |
