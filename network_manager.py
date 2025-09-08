# network_manager.py  — compatible websockets >= 11 (handler(ws) sin 'path')
import asyncio
import json
import threading
import websockets


class NetManager:
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = asyncio.new_event_loop()
        self.server = None
        self._thread = None

    async def _handle_ws(self, ws):
        """Maneja una conexión; compatible con websockets >= 11 (sin path)."""
        self.clients.add(ws)
        try:
            # Mantén viva la conexión; consume si el cliente envía algo
            async for _ in ws:
                pass
        finally:
            self.clients.discard(ws)

    async def _start_async(self):
        # Creamos un wrapper con la firma exacta que espera websockets >= 11
        async def handler(websocket):
            await self._handle_ws(websocket)

        # Si necesitas opciones TLS/headers, agrégalas aquí
        self.server = await websockets.serve(handler, self.host, self.port)
        print(f"[WS] Listening on ws://{self.host}:{self.port}")

    def start(self):
        """Arranca el servidor en un hilo y deja corriendo el event loop."""
        if self._thread is not None:
            return

        def runner():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._start_async())
            self.loop.run_forever()

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()

    async def _broadcast(self, text: str):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def push_state(self, message):
        """Acepta dict/list/str y envía exactamente un JSON texto (una sola vez)."""
        if isinstance(message, (dict, list)):
            text = json.dumps(message, separators=(",", ":"))
        elif isinstance(message, str):
            text = message
        else:
            text = json.dumps(message, default=str, separators=(",", ":"))

        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(text), self.loop)
