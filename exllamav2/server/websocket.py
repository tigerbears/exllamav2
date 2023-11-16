
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

import exllamav2.server.websocket_actions as actions

import websockets, asyncio
import json

class ExLlamaV2WebSocketServer:

    ip: str
    port: int

    model: ExLlamaV2
    draft_model: ExLlamaV2
    tokenizer: ExLlamaV2Tokenizer
    cache: ExLlamaV2Cache
    draft_cache: ExLlamaV2Cache
    generator = ExLlamaV2StreamingGenerator

    def __init__(self, ip: str, port: int, model: ExLlamaV2, tokenizer: ExLlamaV2Tokenizer, cache: ExLlamaV2Cache, draft_model: ExLlamaV2 = None, draft_cache: ExLlamaV2Cache = None):

        self.ip = ip
        self.port = port
        self.model = model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.cache = cache
        self.draft_cache = draft_cache

        self.generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer, draft_model, draft_cache)

    def serve(self):

        print(f" -- Starting WebSocket server on {self.ip} port {self.port}")

        start_server = websockets.serve(self.main, self.ip, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    async def main(self, websocket, path):

        async for message in websocket:

            request = json.loads(message)
            await actions.dispatch(request, websocket, self)