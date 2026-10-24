"""Живая лента событий для интерфейса оператора через WebSocket."""
from fastapi import WebSocket


class LiveManager:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:  # клиент отвалился — убираем его
                self.clients.discard(ws)


manager = LiveManager()
