from fastapi import APIRouter

from ..bots import available_bots

router = APIRouter(prefix="/bots", tags=["bots"])


@router.get("")
async def list_bots():
    """All registered bot names with availability (neural bots need torch + a checkpoint)."""
    info = available_bots()
    return {
        "bots": [{"name": n, **v} for n, v in info.items()],
        "available": [n for n, v in info.items() if v["available"]],
    }
