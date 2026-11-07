"""Game API: moves, bot replies, takeback, resignation, timeout, hints."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def new_game(client, white="human", black="random"):
    r = client.post("/games", json={"white_player": white, "black_player": black})
    assert r.status_code == 200
    return r.json()


def test_move_bot_reply_and_takeback(client):
    g = new_game(client)
    s = client.post(f"/games/{g['id']}/move", json={"move_uci": "e2e4"}).json()
    s = client.post(f"/games/{g['id']}/bot-move").json()
    assert len(s["move_history"]) == 2 and s["turn"] == "white"

    s = client.post(f"/games/{g['id']}/undo", json={"plies": 2}).json()
    assert s["move_history"] == [] and s["fen"].startswith("rnbqkbnr/pppppppp")
    # The DB agrees after the board is reloaded from it.
    from app.game_manager import game_manager
    game_manager.remove(g["id"])
    assert client.get(f"/games/{g['id']}").json()["move_history"] == []


def test_undo_needs_moves(client):
    g = new_game(client)
    assert client.post(f"/games/{g['id']}/undo", json={"plies": 1}).status_code == 400


def test_checkmate_then_takeback(client):
    g = new_game(client, black="human")
    for m in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        s = client.post(f"/games/{g['id']}/move", json={"move_uci": m}).json()
    assert s["is_game_over"] and s["result"] == "0-1" and s["termination"] == "checkmate"
    assert s["legal_moves"] == []
    s = client.post(f"/games/{g['id']}/undo", json={"plies": 1}).json()
    assert not s["is_game_over"] and s["result"] is None


def test_resign_and_timeout(client):
    g = new_game(client)
    s = client.post(f"/games/{g['id']}/resign", json={"color": "white"}).json()
    assert s["is_game_over"] and s["result"] == "0-1" and s["termination"] == "resignation"
    assert client.post(f"/games/{g['id']}/move", json={"move_uci": "e2e4"}).status_code == 400
    assert client.post(f"/games/{g['id']}/bot-move").status_code == 400

    g = new_game(client)
    s = client.post(f"/games/{g['id']}/resign", json={"color": "black", "reason": "timeout"}).json()
    assert s["result"] == "1-0" and s["termination"] == "timeout"


def test_threefold_repetition_ends_game(client):
    g = new_game(client, black="human")
    for m in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        s = client.post(f"/games/{g['id']}/move", json={"move_uci": m}).json()
    assert s["is_game_over"] and s["result"] == "1/2-1/2" and s["termination"] == "repetition"


def test_hint_does_not_move(client):
    g = new_game(client)
    h = client.get(f"/games/{g['id']}/hint", params={"bot": "ab1"}).json()
    assert h["bot"] == "ab1" and h["move_uci"] in g["legal_moves"]
    assert client.get(f"/games/{g['id']}").json()["move_history"] == []
