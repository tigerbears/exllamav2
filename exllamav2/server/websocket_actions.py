
# from exllamav2 import (
#     ExLlamaV2,
#     ExLlamaV2Config,
#     ExLlamaV2Cache,
#     ExLlamaV2Tokenizer
# )

import json

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

async def dispatch(request, ws, server):

    action_ = request["action"]

    response = { "action": action_ }
    if "request_id" in request: response["request_id"] = request["request_id"]
    if "response_id" in request: response["response_id"] = request["response_id"]

    if action_ == "echo": echo(request, ws, server, response)
    elif action_ == "estimate_token": estimate_token(request, ws, server, response)
    elif action_ == "lefttrim_token": lefttrim_token(request, ws, server, response)
    elif action_ == "infer": await infer(request, ws, server, response)

    else:
        print(f" ## Unknown request from client: {request}")
        return

    await ws.send(json.dumps(response))


def echo(request, ws, server, response):

    """
    request:  { action: str = "echo",
                request_id: str,                    # (optional) request ID to echo in response packet
                response_id: str }                  # (optional) response ID to echo in response packet

    response: { action: str = "echo",
                request_id: str,                    # (optional)
                response_id: str }                  # (optional)
    """

    pass


def estimate_token(request, ws, server, response):

    """
    request:  { action: str = "estimate_token",
                request_id: str,                    # (optional) request ID to echo in response packet
                response_id: str,                   # (optional) response ID to echo in response packet
                text: str }                         # text to measure

    response: { action: str = "estimate_token",
                request_id: str,                    # (optional)
                response_id: str,                   # (optional)
                num_tokens: int }                   # length of input text, in tokens
    """

    text = request["text"]
    ids = server.tokenizer.cached_encode_str(text)
    response["num_tokens"] = ids.shape[-1]


def lefttrim_token(request, ws, server, response):

    """
    request:  { action: str = "lefttrim_token",
                request_id: str,                    # (optional) request ID to echo in response packet
                response_id: str,                   # (optional) response ID to echo in response packet
                text: str,                          # text to trim
                trimmed_length: int }               # num tokens to keep, from right

    response: { action: str = "lefttrim_token",
                request_id: str,                    # (optional)
                response_id: str,                   # (optional)
                trimmed_text: str }                 # input, trimmed
    """

    text = request["text"]
    length = int(request["trimmed_length"])

    ids = server.tokenizer.cached_encode_str(text)
    if ids.shape[-1] <= length:
        response["trimmed_text"] = text
    else:
        response["trimmed_text"] = server.tokenizer.decode(ids[:, -length:])[0]


async def infer(request, ws, server, response):

    """
    request:  { action: str = "infer",
                request_id: str,                    # (optional) request ID to echo in response packet
                response_id: str,                   # (optional) response ID to echo in response packet
                text: str,                          # input prompt
                max_new_tokens: int,                # max num new tokens
                stream: bool,                       # stream response
                stream_full: bool,                  # return full response-so-far with each streamed chunk

                min_p: float,                       # (optional) min-P threshold (0 to disable)
                mirostat: bool,                     # (optional) enable mirostat sampling
                mirostat_eta: float,                # (optional) mirostat_eta, ranges from 0-1
                mirostat_tau: float,                # (optional) mirostat_tau, ranges from 0-10
                rep_pen: float,                     # (optional) repetition penalty (1.0 = no penalty)
                rep_pen_decay: int,                 # (optional) repetition penalty decay
                rep_pen_range: int,                 # (optional) repetition penalty range, in tokens
                temperature: float,                 # (optional) sampling temperature (1.0 = no temp adjust)
                tfs: float,                         # (optional) tail-free sampling (0 to disable)
                top_k: int,                         # (optional) top-K count (0 to disable)
                top_p: float,                       # (optional) top-P threshold (0 to disable)
                typical: float,                     # (optional) typical threshold (0 to disable)

                stop_conditions: [str|int],         # (optional) list of stop conditions
                token_healing: bool,                # (optional) enable token healing
                tag: str }                          # (optional) tag to echo in response packet

    streams:  { action: str = "infer",
                request_id: str,                    # (optional)
                response_id: str,                   # (optional)
                response_type: str = "chunk",
                chunk: str,                         # next chunk of response
                tag: str }                          # (optional)

    response: { action: str = "infer",
                request_id: str,                    # (optional)
                response_id: str,                   # (optional)
                response_type: str = "full",
                util_text: str,                     # input context (pruned if max_seq_len exceeded)
                response: str,                      # full response excluding input prompt
                tag: str }                          # (optional)
    """

    # Mode

    stream = request["stream"]
    if "tag" in request:
        response["tag"] = request["tag"]

    # Stop conditions

    sc = [server.tokenizer.eos_token_id]
    if "stop_conditions" in request:
        ss = request["stop_conditions"]
        if not isinstance(ss, list): ss = [ss]
        sc += ss

    # Full response

    full_response = request.get("full_response", False)

    # Tokenize and trim prompt

    full_ctx = request["text"]
    num_tokens = request["max_new_tokens"]

    ids = server.tokenizer.cached_encode_str(full_ctx)
    overflow = ids.shape[-1] + num_tokens - server.model.config.max_seq_len
    if overflow > 0:
        ids = ids[:, overflow:]
        util_ctx = server.tokenizer.decode(ids)
    else:
        util_ctx = full_ctx

    # Sampler

    gs = ExLlamaV2Sampler.Settings()
    gs.min_p = float(request["min_p"]) if "min_p" in request else 0
    gs.mirostat = bool(request["mirostat"]) if "mirostat" in request else False
    gs.mirostat_eta = float(request["mirostat_eta"]) if "mirostat_eta" in request else 0.1
    gs.mirostat_tau = float(request["mirostat_tau"]) if "mirostat_tau" in request else 5
    gs.token_repetition_penalty = float(request["rep_pen"]) if "rep_pen" in request else 1.15
    gs.token_repetition_decay = int(request["rep_pen_decay"]) if "rep_pen_decay" in request else 0
    gs.token_repetition_range = int(request["rep_pen_range"]) if "rep_pen_range" in request else -1
    gs.temperature = float(request["temperature"]) if "temperature" in request else 0.95
    gs.tfs = float(request["tfs"]) if "tfs" in request else 0
    gs.top_k = int(request["top_k"]) if "top_k" in request else 100
    gs.top_p = float(request["top_p"]) if "top_p" in request else 0.8
    gs.typical = float(request["typical"]) if "typical" in request else 0

    # Generate

    server.generator.set_stop_conditions(sc)
    server.generator.begin_stream(ids, gs, token_healing = request["token_healing"] if "token_healing" in request else False)

    completion = ""
    gen_tokens = 0

    while True:
        chunk, eos, _ = server.generator.stream()
        completion += chunk
        gen_tokens += 1

        if stream and chunk != "":
            response["response_type"] = "chunk"
            response["chunk"] = chunk
            if full_response: response["response"] = completion
            await ws.send(json.dumps(response))

        if eos or gen_tokens >= num_tokens: break

    if stream and "chunk" in response: del response["chunk"]
    response["response_type"] = "full"
    response["util_text"] = util_ctx
    response["response"] = completion
