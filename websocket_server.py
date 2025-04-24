import asyncio
import json
import logging
from typing import Callable, Set, Union, Optional

import websockets

class WebSocketServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        """Initialize WebSocket server.

        Args:
            host: Host address to bind the server
            port: Port number to bind the server
        """
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self.on_connect_callback: Optional[Callable] = None
        self.on_message_callback: Optional[Callable] = None
        self.on_disconnect_callback: Optional[Callable] = None
        self._is_running = False
        self.logger = logging.getLogger(__name__)

    async def _handler(self, websocket):
        """Handle incoming websocket connections."""
        self.clients.add(websocket)

        # Default path
        path = "/"

        # For websockets library v10+, try to get path from request
        if hasattr(websocket, "request"):
            path = websocket.request.path

        if self.on_connect_callback:
            await self.on_connect_callback(websocket, path)

        try:
            async for message in websocket:
                try:
                    if self.on_message_callback:
                        await self.on_message_callback(websocket, message)
                    else:
                        self.logger.info(f"Received message: {message}")
                except Exception as e:
                    self.logger.error(f"Error handling message: {e}")
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed: {id(websocket)}")
        finally:
            self.clients.remove(websocket)
            if self.on_disconnect_callback:
                await self.on_disconnect_callback(websocket)

    def on_connect(self, callback: Callable):
        """Register a callback for when a client connects."""
        self.on_connect_callback = callback
        return self

    def on_message(self, callback: Callable):
        """Register a callback for when a message is received."""
        self.on_message_callback = callback
        return self

    def on_disconnect(self, callback: Callable):
        """Register a callback for when a client disconnects."""
        self.on_disconnect_callback = callback
        return self

    async def send_to_client(self, client, message: Union[str, dict]):
        """Send a message to a specific client."""
        if isinstance(message, dict):
            message = json.dumps(message)
        await client.send(message)

    async def broadcast(self, message: Union[str, dict]):
        """Send a message to all connected clients."""
        if isinstance(message, dict):
            message = json.dumps(message)
        if self.clients:
            await asyncio.gather(*[client.send(message) for client in self.clients])

    async def start(self):
        """Start the websocket server."""
        if self._is_running:
            self.logger.warning("Server is already running")
            return

        self.server = await websockets.serve(self._handler, self.host, self.port)
        self._is_running = True
        self.logger.info(f"Server started on ws://{self.host}:{self.port}")
        return self

    async def stop(self):
        """Stop the websocket server."""
        if not self._is_running:
            self.logger.warning("Server is not running")
            return

        # Close all client connections
        if self.clients:
            await asyncio.gather(*[client.close() for client in self.clients])
            self.clients.clear()

        # Stop the server
        self.server.close()
        await self.server.wait_closed()

        self._is_running = False
        self.logger.info(f"Server stopped on ws://{self.host}:{self.port}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    server = WebSocketServer()

    async def on_connect(websocket, path):
        print(f"Client connected: {id(websocket)}")

    async def on_message(websocket, message):
        print(f"Message from {id(websocket)}: {message}")
        await server.send_to_client(websocket, {"response": "Message received"})

    async def on_disconnect(websocket):
        print(f"Client disconnected: {id(websocket)}")

    server.on_connect(on_connect).on_message(on_message).on_disconnect(on_disconnect)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(server.start())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.stop())
        loop.close()