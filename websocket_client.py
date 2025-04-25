import asyncio
import websockets


async def mock_websocket_client():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Continuously send messages to the server
        messages = ["Annotation 1", "Annotation 2", "Annotation 3"]
        while True:  # Infinite loop to keep sending messages
            for message in messages:
                await websocket.send(message)
                response = await websocket.recv()
                print(f"Server response: {response}")
                await asyncio.sleep(1)  # Simulate delay between messages


# Run the simulation
if __name__ == "__main__":
    asyncio.run(mock_websocket_client())
