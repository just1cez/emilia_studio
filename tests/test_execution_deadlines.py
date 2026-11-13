"""Graph deadlines exercised through a real local HTTP/WebSocket server."""
import asyncio
from dataclasses import replace
from pathlib import Path

import aiohttp
from aiohttp import web
import pytest

from emilia_studio.config import load_config
from emilia_studio.engine import EmiliaRenderEngine, StageServer, run_is_active, run_lock
from emilia_studio.inspector import QualityInspector
from emilia_studio.ledger import ExperimentLedger


@pytest.mark.parametrize("mode", ["stall", "unrelated", "progress", "submission", "history"])
def test_graph_deadlines_release_websocket_and_run_lock(tmp_path, mode):
    async def scenario():
        clients = set()
        accepted = asyncio.Event()
        disconnected = asyncio.Event()

        async def websocket(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            clients.add(ws)
            accepted.set()
            try:
                async for _ in ws:
                    pass
            finally:
                clients.discard(ws)
                disconnected.set()
            return ws

        async def prompt(request):
            await request.json()
            if mode == "submission":
                await asyncio.sleep(0.5)
            if mode == "history":
                for ws in list(clients):
                    await ws.send_json({"type": "executing", "data": {"prompt_id": "pid", "node": None}})
            return web.json_response({"prompt_id": "pid"})

        async def history(request):
            await asyncio.sleep(0.5)
            return web.json_response({"pid": {"status": {"status_str": "success"}, "outputs": {}}})

        async def noise():
            await accepted.wait()
            while True:
                for ws in list(clients):
                    if mode == "unrelated":
                        await ws.send_json({"type": "progress", "data": {"prompt_id": "other", "value": 1, "max": 2}})
                        await ws.send_bytes(b"unrelated preview")
                    elif mode == "progress":
                        await ws.send_json({"type": "progress", "data": {"prompt_id": "pid", "value": 1, "max": 2}})
                await asyncio.sleep(0.02)

        app = web.Application()
        app.router.add_get("/ws", websocket)
        app.router.add_post("/prompt", prompt)
        app.router.add_get("/history/{pid}", history)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        config = replace(load_config(), root_dir=tmp_path, app_dir=tmp_path / "app",
                         stage_timeout_s=0.6 if mode in ("stall", "unrelated") else 0.2,
                         stall_timeout_s=0.15 if mode in ("stall", "unrelated") else 0)
        engine = EmiliaRenderEngine(config=config, inspector=QualityInspector(use_yolo=False),
                                    ledger=ExperimentLedger(tmp_path / "ledger.sqlite3", config))
        noise_task = asyncio.create_task(noise())
        try:
            async with aiohttp.ClientSession() as session:
                server = StageServer(session, f"http://127.0.0.1:{port}", tmp_path, tmp_path, None)
                pattern = "stalled" if mode in ("stall", "unrelated") else "stage_timeout_s"
                with pytest.raises(TimeoutError, match=pattern):
                    with run_lock(tmp_path):
                        await engine._execute_graph(server, graph={}, stage="native")
                await asyncio.wait_for(disconnected.wait(), timeout=2)
                assert not run_is_active(tmp_path)
        finally:
            noise_task.cancel()
            await asyncio.gather(noise_task, return_exceptions=True)
            await runner.cleanup()
    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["stage_timeout_s", "stall_timeout_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_timeout_rejected_before_launch(field, value):
    with pytest.raises(ValueError, match=field):
        replace(load_config(), **{field: value})
