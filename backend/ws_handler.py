import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from gesture_api import GestureEngine


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_text(json.dumps(data))


manager = ConnectionManager()


async def gesture_ws_handler(websocket: WebSocket):
    """
    WebSocket endpoint handler.

    Protocol:
      Browser → sends raw JPEG bytes (binary message)
      Server  → sends JSON string with gesture result

    JSON response format:
    {
        "direction":  "UP" | "DOWN" | "LEFT" | "RIGHT" | null,
        "special":    "FIST" | "PALM" | null,
        "right_hand": true/false,
        "left_hand":  true/false
    }
    """
    await manager.connect(websocket)
    engine = GestureEngine()

    try:
        while True:
            # Receive binary frame from browser
            frame_bytes = await websocket.receive_bytes()

            # Process frame in gesture engine
            result = engine.process_frame(frame_bytes)

            # Send result back to browser
            await manager.send_json(websocket, result)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        engine.release()

    except Exception as e:
        print(f"[WS] Error: {e}")
        manager.disconnect(websocket)
        engine.release()