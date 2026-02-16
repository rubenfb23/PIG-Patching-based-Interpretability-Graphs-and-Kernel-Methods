"""Websocket API for local graph viewer."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from websockets.asyncio.server import serve

from pig.web.schemas import ViewFilter, build_message
from pig.web.service import (
    GraphViewerService,
    ViewerConfig,
    load_dataset_from_cache,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pig-viewer",
        description="PIG Viewer backend",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/patch_effects",
        help="Directory with PatchEffectCache JSON files",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host",
    )
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k edges per node",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=3000,
        help="Default maximum number of edges returned",
    )
    return parser.parse_args()


async def _handle_connection(websocket, service: GraphViewerService) -> None:
    await websocket.send(
        json.dumps(
            build_message("init", payload=service.initial_payload()),
        )
    )

    async for raw_message in websocket:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            await websocket.send(
                json.dumps(
                    build_message(
                        "error",
                        message="Invalid JSON payload",
                    )
                )
            )
            continue

        message_type = str(message.get("type", "")).strip().lower()
        if message_type == "ping":
            await websocket.send(json.dumps(build_message("pong")))
            continue

        if message_type != "filter_update":
            await websocket.send(
                json.dumps(
                    build_message(
                        "error",
                        message=f"Unsupported message type '{message_type}'",
                    )
                )
            )
            continue

        try:
            view_filter = ViewFilter(
                slice_id=str(message["slice_id"]),
                layer_min=message.get("layer_min"),
                layer_max=message.get("layer_max"),
                token_min=message.get("token_min"),
                token_max=message.get("token_max"),
                min_abs_weight=float(message.get("min_abs_weight", 0.0)),
                max_edges=int(message.get("max_edges", 3000)),
            )
            graph_payload = service.filtered_graph(view_filter)
        except (KeyError, TypeError, ValueError) as exc:
            await websocket.send(
                json.dumps(
                    build_message(
                        "error",
                        message=str(exc),
                    )
                )
            )
            continue

        await websocket.send(
            json.dumps(
                build_message(
                    "graph_update",
                    payload=graph_payload,
                )
            )
        )


async def _run_server(args: argparse.Namespace) -> None:
    dataset = load_dataset_from_cache(Path(args.cache_dir))
    service = GraphViewerService(
        dataset=dataset,
        config=ViewerConfig(
            top_k=args.top_k,
            max_edges_default=args.max_edges,
        ),
    )

    async with serve(
        lambda websocket: _handle_connection(websocket, service),
        args.host,
        args.port,
    ):
        print(f"Viewer backend listening on ws://{args.host}:{args.port}")
        await asyncio.Future()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run_server(args))
    except ValueError as exc:
        print(f"Viewer backend startup error: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
