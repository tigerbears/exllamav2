
import argparse, sys, os, glob

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
    parser.add_argument("-rs", "--rope_scale", type = float, default = 1.0, help = "RoPE scaling factor")
    parser.add_argument("-ra", "--rope_alpha", type = float, default = 1.0, help = "RoPE alpha value (NTK)")
    parser.add_argument("-nfa", "--no_flash_attn", action = "store_true", help = "Disable Flash Attention")
    parser.add_argument("-lm", "--low_mem", action = "store_true", help = "Enable VRAM optimizations, potentially trading off speed")


def print_options(args):

    print(f" -- Model: {args.model_dir}")

    print_opts = []
    if args.draft_model_dir: print_opts += [f"draft_model_dir: {args.draft_model_dir}"]
    if args.no_draft_scale: print_opts += [f"no_draft_scale: {args.no_draft_scale}"]
    if args.gpu_split: print_opts += [f"gpu_split: {args.gpu_split}"]
    if args.length: print_opts += [f"length: {args.length}"]
    print_opts += [f"rope_scale {args.rope_scale}"]
    print_opts += [f"rope_alpha {args.rope_alpha}"]
    if args.no_flash_attn: print_opts += ["no_flash_attn"]
    if args.low_mem: print_opts += ["low_mem"]
    print(f" -- Options: {print_opts}")


def check_model_directory(directory):

    if not os.path.exists(directory):
        print(f" ## Error: Can't find directory provided in arguments: {directory}")
        sys.exit()

    required_files = ["config.json",
                      "tokenizer.model",
                      "*.safetensors"]

    for filename in required_files:

        path = os.path.join(directory, filename)
        matches = glob.glob(path)
        if len(matches) == 0:
            print(f" ## Error: Cannot find {filename} in {directory}")
            sys.exit()


def check_args(args):

    if not args.model_dir:
        print(" ## Error: No model directory specified")
        sys.exit()

    check_model_directory(args.model_dir)

    if args.draft_model_dir is not None:
        check_model_directory(args.draft_model_dir)


def config_for_model_directory(args, directory, draft_of_config = None):

    if directory is None:
        return None

    # Create config

    config = ExLlamaV2Config()
    config.model_dir = directory
    config.prepare()

    if args.length: config.max_seq_len = args.length

    config.scale_pos_emb = args.rope_scale
    config.scale_alpha_value = args.rope_alpha
    config.no_flash_attn = args.no_flash_attn

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

    return config


def init(args, quiet = False, allow_auto_split = False):

    # Create config

    model_config = config_for_model_directory(args, args.model_dir)
    draft_config = config_for_model_directory(args, args.draft_model_dir, model_config)

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

    if args.gpu_split != "auto":
        if not quiet: print(" -- Loading model...")
        model.load(split)

        if draft_model is not None:
            if not quiet: print(" -- Loading draft model...")
            draft_model.load(split)

    else:
        assert allow_auto_split, "Auto split not allowed."

    # Load tokenizer

    if not quiet: print(" -- Loading tokenizer...")

    tokenizer = ExLlamaV2Tokenizer(model_config)

    return model, draft_model, tokenizer