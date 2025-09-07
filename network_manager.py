import asyncio
import websockets
import websockets.exceptions
import json
from typing import Any

class NetManager:
    def __init__(self, host="localhost", port=8080):
        self.host = host
        self.port = port
        self.clients: set[Any] = set()
        self.server = None
        self.loop = None

    async def _start_server(self):
        self.server = await websockets.serve(self._handler, self.host, self.port)
        print(f"WebSocket server started at ws://{self.host}:{self.port}")
        await self._heartbeat()  # This will keep the server running

    def start(self):
        """Start the websocket server in a background thread."""
        import threading
        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._start_server())
        t = threading.Thread(target=run, daemon=True)
        t.start()

    # websockets>=10 passes only the websocket, older versions passed (websocket, path)
    async def _handler(self, websocket):
        self.clients.add(websocket)
        print(f"Client connected: {websocket.remote_address}")
        try:
            # Send a welcome message to confirm connection
            await websocket.send(json.dumps({"type": "welcome", "message": "Connected to simulation server"}))
            
            # Keep the connection alive by waiting for messages or connection close
            async for message in websocket:
                # Handle incoming messages from client (optional)
                try:
                    data = json.loads(message)
                    print(f"Received message from {websocket.remote_address}: {data}")
                    # You can add message handling logic here if needed
                except json.JSONDecodeError:
                    print(f"Received non-JSON message from {websocket.remote_address}: {message}")
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed by client: {websocket.remote_address}")
        except Exception as e:
            print(f"Error handling client {websocket.remote_address}: {e}")
        finally:
            self.clients.discard(websocket)
            print(f"Client disconnected: {websocket.remote_address}")

    async def _heartbeat(self):
        """Keep alive loop, can also be used for pings"""
        while True:
            await asyncio.sleep(1)

    def push_state(self, state):
        """Serialize and send state to all connected clients"""
        if not self.clients:
            return # No clients connected

        message = json.dumps(state)
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    async def _broadcast(self, message):
        dead_clients = []
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                dead_clients.append(client)

        for dc in dead_clients:
            self.clients.remove(dc)
            print(f"Removed dead client: {dc.remote_address}")