from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self._connections: dict[int, list[WebSocket]] = {}

    async def connect(self, game_id: int, ws: WebSocket):
        await ws.accept()
        self._connections.setdefault(game_id, []).append(ws)

    def disconnect(self, game_id: int, ws: WebSocket):
        conns = self._connections.get(game_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, game_id: int, data: dict):
        conns = list(self._connections.get(game_id, []))
        dead = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(game_id, ws)


connection_manager = ConnectionManager()
