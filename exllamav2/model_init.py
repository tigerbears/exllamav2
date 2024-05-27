
import argparse, sys, os, glob, time

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer
)


def add_args(parser):
    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory")
    parser.add_argument("-dm", "--draft_model_dir", type = str, help = "Path to draft model directory for speculative decoding")
    parser.add_argument("-nds", "--no_draft_scale", action = "store_true", help = "If draft model has smaller context size than model, don't apply alpha (NTK) scaling to extend it")
    parser.add_argument("-gs", "--gpu_split", type = str, help = "\"auto\", or VRAM allocation per GPU in GB")
    parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length")
    parser.add_argument("-rs", "--rope_scale", type = float, help = "RoPE scaling factor")
    parser.add_argument("-ra", "--rope_alpha", type = float, help = "RoPE alpha value (NTK)")
    parser.add_argument("-nfa", "--no_flash_attn", action = "store_true", help = "Disable Flash Attention")
    parser.add_argument("-lm", "--low_mem", action = "store_true", help = "Enable VRAM optimizations, potentially trading off speed")
    parser.add_argument("-ept", "--experts_per_token", type = int, help = "Override MoE model's default number of experts per token")
    parser.add_argument("-lq4", "--load_q4", action = "store_true", help = "Load weights in Q4 mode")
    if os.name != "nt":
        parser.add_argument("-fst", "--fast_safetensors", action = "store_true", help = "Optimized safetensors loading with direct I/O (experimental!)")


def print_options(args):

    print(f" -- Model: {args.model_dir}")

    print_opts = []
    if args.gpu_split is not None: print_opts += [f"gpu_split: {args.gpu_split}"]
    if args.length is not None: print_opts += [f"length: {args.length}"]
    if args.rope_scale is not None: print_opts += [f"rope_scale: {args.rope_scale}"]
    if args.rope_alpha is not None: print_opts += [f"rope_alpha: {args.rope_alpha}"]

    if args.draft_model_dir is not None: print_opts += [f"draft_model_dir: {args.draft_model_dir}"]
    if args.no_draft_scale: print_opts += [f"no_draft_scale: {args.no_draft_scale}"]

    if args.no_flash_attn: print_opts += ["no_flash_attn"]
    if args.low_mem: print_opts += ["low_mem"]
    if hasattr(args, "fast_safetensors") and args.fast_safetensors: print_opts += ["fast_safetensors"]
    if args.experts_per_token is not None: print_opts += [f"experts_per_token: {args.experts_per_token}"]
    if args.load_q4: print_opts += ["load_q4"]
    print(f" -- Options: {print_opts}")


def check_model_directory(directory):

    if not os.path.exists(directory):
        print(f" ## Error: Can't find directory provided in arguments: {directory}")
        sys.exit()

    required_files = ["config.json",
                      ["tokenizer.model", "tokenizer.json"],
                      "*.safetensors"]

    for filename in required_files:
        if isinstance(filename, str):
            filename = [filename]
        all_matches = []
        for file in filename:
            path = os.path.join(directory, file)
            matches = glob.glob(path)
            all_matches += matches
        if len(all_matches) == 0:
            print(f" ## Error: Cannot find {filename} in {directory}")
            sys.exit()


def check_args(args):

    if not args.model_dir:
        print(" ## Error: No model directory specified")
        sys.exit()

    check_model_directory(args.model_dir)

    if args.draft_model_dir is not None:
        check_model_directory(args.draft_model_dir)


def config_for_model_directory(args, directory, max_batch_size: int = None, max_output_len: int = None, draft_of_config = None):

    if directory is None:
        return None

    # Create config

    config = ExLlamaV2Config()
    config.model_dir = directory
    config.fasttensors = hasattr(args, "fast_safetensors") and args.fast_safetensors
    config.prepare()

    if args.length: config.max_seq_len = args.length
    if args.rope_scale: config.scale_pos_emb = args.rope_scale
    if args.rope_alpha: config.scale_alpha_value = args.rope_alpha
    config.no_flash_attn = args.no_flash_attn
    if args.experts_per_token: config.num_experts_per_token = args.experts_per_token

    if max_batch_size: config.max_batch_size = max_batch_size
    config.max_output_len = max_output_len

    # Set config options
    if draft_of_config is not None:

        if config.max_seq_len < draft_of_config.max_seq_len:

            if args.no_draft_scale:
                print(f" !! Warning: Draft model native max sequence length is less than sequence length for model. Speed may decrease after {config.max_seq_len} tokens.")
            else:
                ratio = draft_of_config.max_seq_len / config.max_seq_len
                alpha = -0.13436 + 0.80541 * ratio + 0.28833 * ratio ** 2
                config.scale_alpha_value = alpha
                config.max_seq_len = draft_of_config.max_seq_len
                print(f" -- Applying draft model RoPE alpha = {alpha:.4f}")

    # Set low-mem options

    if args.low_mem: config.set_low_mem()
    if args.load_q4: config.load_in_q4 = True

    return config


def init(args,
         quiet: bool = False,
         allow_auto_split: bool = False,
         skip_load: bool = False,
         benchmark: bool = False,
         max_batch_size: int = None,
         max_output_len: int = None):

    # Create config

    model_config = config_for_model_directory(args, args.model_dir, max_batch_size, max_output_len)
    draft_config = config_for_model_directory(args, args.draft_model_dir, max_batch_size, max_output_len, model_config)

    # Load model
    # If --gpu_split auto, return unloaded model. Model must be loaded with model.load_autosplit() supplying cache
    # created in lazy mode

    model = ExLlamaV2(model_config)
    draft_model = None
    if draft_config is not None:
        draft_model = ExLlamaV2(draft_config)

    split = None
    if args.gpu_split and args.gpu_split != "auto":
        split = [float(alloc) for alloc in args.gpu_split.split(",")]

    if args.gpu_split != "auto" and not skip_load:
        if not quiet: print(" -- Loading model...")
        t = time.time()
        model.load(split)
        t = time.time() - t
        if benchmark and not quiet:
            print(f" -- Loaded model in {t:.4f} seconds")

        if draft_model is not None:
            if not quiet: print(" -- Loading draft model...")
            t = time.time()
            draft_model.load(split)
            t = time.time() - t
            if benchmark and not quiet:
                print(f" -- Loaded draft model in {t:.4f} seconds")

    else:
        assert allow_auto_split, "Auto split not allowed."

    # Load tokenizer

    if not quiet: print(" -- Loading tokenizer...")

    tokenizer = ExLlamaV2Tokenizer(model_config)

    return model, draft_model, tokenizer