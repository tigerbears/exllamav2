
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2Cache, model_init
from exllamav2.server import ExLlamaV2WebSocketServer

import argparse

# Configure and init model

parser = argparse.ArgumentParser(description = "WebSocket server example")
parser.add_argument("-host", "--host", type = str, default = "0.0.0.0:7862", help = "IP:PORT eg, 0.0.0.0:7862")

model_init.add_args(parser)
args = parser.parse_args()
model_init.check_args(args)
model_init.print_options(args)
model, draft_model, tokenizer = model_init.init(args, allow_auto_split = True)
draft_cache = None

# Load model after cache if --gpu_split auto

if not model.loaded:
    cache = ExLlamaV2Cache(model, lazy = True)
    model.load_autosplit(cache)

# Else create cache

else:
    cache = ExLlamaV2Cache(model)

# Do the same for the draft model, if it exists

if draft_model is not None:

    # Load model after cache if --gpu_split auto

    if not draft_model.loaded:
        draft_cache = ExLlamaV2Cache(draft_model, lazy = True)
        draft_model.load_autosplit(cache)

    # Else create cache

    else:
        draft_cache = ExLlamaV2Cache(draft_model)

# Create server

ip, port = args.host.split(":")
port = int(port)

server = ExLlamaV2WebSocketServer(ip, port, model, tokenizer, cache, draft_model, draft_cache)
server.serve()
