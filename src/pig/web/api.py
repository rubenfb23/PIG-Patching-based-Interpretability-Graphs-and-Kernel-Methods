"""Websocket API for local graph viewer."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from websockets.asyncio.server import serve

from pig.web.schemas import ViewFilter, build_message
from pig.web.service import (
    GraphViewerService,
    ViewerConfig,
    load_dataset_from_cache,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pig-viewer",
        description="PIG Viewer backend",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/patch_effects",
        help=(
            "Directory with PatchEffectCache JSON files, or a parent "
            "directory with per-run subdirectories (latest run auto-selected)"
        ),
    )
    parser.add_argument(
        "--base-cache-dir",
        default=None,
        help="Cache dir del modelo base. Activa diff_correlation_topk.",
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
    client = getattr(websocket, "remote_address", None)
    logger.info("[viewer][ws] client connected: %s", client)
    init_payload = service.initial_payload()
    logger.info(
        "[viewer][ws] sending init to %s: slices=%d nodes=%d edges=%d",
        client,
        len(init_payload.get("slices", [])),
        len(init_payload.get("graph", {}).get("nodes", [])),
        len(init_payload.get("graph", {}).get("edges", [])),
    )
    await websocket.send(
        json.dumps(
            build_message("init", payload=init_payload),
        )
    )

    async for raw_message in websocket:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.warning("[viewer][ws] invalid json from %s", client)
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
        logger.info("[viewer][ws] recv %s from %s", message_type or "<empty>", client)
        if message_type == "ping":
            await websocket.send(json.dumps(build_message("pong")))
            continue

        if message_type != "filter_update":
            logger.warning(
                "[viewer][ws] unsupported message type '%s' from %s",
                message_type,
                client,
            )
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
                max_edges=int(message.get("max_edges", 1000)),
            )
            graph_payload = service.filtered_graph(view_filter)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("[viewer][ws] bad filter_update from %s: %s", client, exc)
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
        logger.info(
            "[viewer][ws] sent graph_update to %s: nodes=%d edges=%d slice=%s",
            client,
            len(graph_payload.get("nodes", [])),
            len(graph_payload.get("edges", [])),
            graph_payload.get("slice", ""),
        )


async def _run_server(args: argparse.Namespace) -> None:
    logger.info("[viewer] loading dataset from %s", args.cache_dir)
    dataset = load_dataset_from_cache(Path(args.cache_dir))
    builder = None
    if args.base_cache_dir:
        logger.info("[viewer] diff mode enabled with base cache %s", args.base_cache_dir)
        base_dataset = load_dataset_from_cache(Path(args.base_cache_dir))
        from pig.graphs.diff_correlation_topk import DiffCorrelationGraphBuilder

        builder = DiffCorrelationGraphBuilder(
            base_dataset=base_dataset,
            k=args.top_k,
        )

    service = GraphViewerService(
        dataset=dataset,
        config=ViewerConfig(
            top_k=args.top_k,
            max_edges_default=args.max_edges,
        ),
        builder=builder,
    )

    async with serve(
        lambda websocket: _handle_connection(websocket, service),
        args.host,
        args.port,
    ):
        print(f"Viewer backend listening on ws://{args.host}:{args.port}")
        await asyncio.Future()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    try:
        asyncio.run(_run_server(args))
    except ValueError as exc:
        print(f"Viewer backend startup error: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
