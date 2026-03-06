from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
from ws_handler import gesture_ws_handler

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Gesture Snake AI - Backend",
    description="Real-time hand gesture detection via MediaPipe and WebSockets",
    version="1.0.0"
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allow frontend (Vercel / localhost) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten this in production to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── HTTP Routes ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Gesture Snake AI Backend is running 🐍",
        "websocket": "ws://<host>/ws/gesture"
    }

@app.get("/health")
async def health():
    """Health check endpoint — used by Render/Railway to verify the service."""
    return {"status": "healthy"}


# ─── WebSocket Route ──────────────────────────────────────────────────────────
@app.websocket("/ws/gesture")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time gesture detection.

    Flow:
      1. Browser connects to ws://<host>/ws/gesture
      2. Browser sends JPEG frame bytes every ~100ms
      3. Server runs MediaPipe, returns JSON gesture result
      4. Browser updates snake direction based on result
    """
    await gesture_ws_handler(websocket)


# ─── Run locally ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)