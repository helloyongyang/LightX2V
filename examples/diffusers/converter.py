import os
import re
import glob
import json
import argparse
import torch
from safetensors import safe_open, torch as st
from loguru import logger
from tqdm import tqdm


def get_key_mapping_rules(direction, model_type):
    if model_type == "wan":
        unified_rules = [
            {"forward": (r"^head\.head$", "proj_out"), "backward": (r"^proj_out$", "head.head")},
            {"forward": (r"^head\.modulation$", "scale_shift_table"), "backward": (r"^scale_shift_table$", "head.modulation")},
            {"forward": (r"^text_embedding\.0\.", "condition_embedder.text_embedder.linear_1."), "backward": (r"^condition_embedder.text_embedder.linear_1\.", "text_embedding.0.")},
            {"forward": (r"^text_embedding\.2\.", "condition_embedder.text_embedder.linear_2."), "backward": (r"^condition_embedder.text_embedder.linear_2\.", "text_embedding.2.")},
            {"forward": (r"^time_embedding\.0\.", "condition_embedder.time_embedder.linear_1."), "backward": (r"^condition_embedder.time_embedder.linear_1\.", "time_embedding.0.")},
            {"forward": (r"^time_embedding\.2\.", "condition_embedder.time_embedder.linear_2."), "backward": (r"^condition_embedder.time_embedder.linear_2\.", "time_embedding.2.")},
            {"forward": (r"^time_projection\.1\.", "condition_embedder.time_proj."), "backward": (r"^condition_embedder.time_proj\.", "time_projection.1.")},
            {"forward": (r"blocks\.(\d+)\.self_attn\.q\.", r"blocks.\1.attn1.to_q."), "backward": (r"blocks\.(\d+)\.attn1\.to_q\.", r"blocks.\1.self_attn.q.")},
            {"forward": (r"blocks\.(\d+)\.self_attn\.k\.", r"blocks.\1.attn1.to_k."), "backward": (r"blocks\.(\d+)\.attn1\.to_k\.", r"blocks.\1.self_attn.k.")},
            {"forward": (r"blocks\.(\d+)\.self_attn\.v\.", r"blocks.\1.attn1.to_v."), "backward": (r"blocks\.(\d+)\.attn1\.to_v\.", r"blocks.\1.self_attn.v.")},
            {"forward": (r"blocks\.(\d+)\.self_attn\.o\.", r"blocks.\1.attn1.to_out.0."), "backward": (r"blocks\.(\d+)\.attn1\.to_out\.0\.", r"blocks.\1.self_attn.o.")},
            {"forward": (r"blocks\.(\d+)\.cross_attn\.q\.", r"blocks.\1.attn2.to_q."), "backward": (r"blocks\.(\d+)\.attn2\.to_q\.", r"blocks.\1.cross_attn.q.")},
            {"forward": (r"blocks\.(\d+)\.cross_attn\.k\.", r"blocks.\1.attn2.to_k."), "backward": (r"blocks\.(\d+)\.attn2\.to_k\.", r"blocks.\1.cross_attn.k.")},
            {"forward": (r"blocks\.(\d+)\.cross_attn\.v\.", r"blocks.\1.attn2.to_v."), "backward": (r"blocks\.(\d+)\.attn2\.to_v\.", r"blocks.\1.cross_attn.v.")},
            {"forward": (r"blocks\.(\d+)\.cross_attn\.o\.", r"blocks.\1.attn2.to_out.0."), "backward": (r"blocks\.(\d+)\.attn2\.to_out\.0\.", r"blocks.\1.cross_attn.o.")},
            {"forward": (r"blocks\.(\d+)\.norm3\.", r"blocks.\1.norm2."), "backward": (r"blocks\.(\d+)\.norm2\.", r"blocks.\1.norm3.")},
            {"forward": (r"blocks\.(\d+)\.ffn\.0\.", r"blocks.\1.ffn.net.0.proj."), "backward": (r"blocks\.(\d+)\.ffn\.net\.0\.proj\.", r"blocks.\1.ffn.0.")},
            {"forward": (r"blocks\.(\d+)\.ffn\.2\.", r"blocks.\1.ffn.net.2."), "backward": (r"blocks\.(\d+)\.ffn\.net\.2\.", r"blocks.\1.ffn.2.")},
            {"forward": (r"blocks\.(\d+)\.modulation\.", r"blocks.\1.scale_shift_table."), "backward": (r"blocks\.(\d+)\.scale_shift_table(?=\.|$)", r"blocks.\1.modulation")},
            {"forward": (r"blocks\.(\d+)\.cross_attn\.k_img\.", r"blocks.\1.attn2.add_k_proj."), "backward": (r"blocks\.(\d+)\.attn2\.add_k_proj\.", r"blocks.\1.cross_attn.k_img.")},
            {"forward": (r"blocks\.(\d+)\.cross_attn\.v_img\.", r"blocks.\1.attn2.add_v_proj."), "backward": (r"blocks\.(\d+)\.attn2\.add_v_proj\.", r"blocks.\1.cross_attn.v_img.")},
            {
                "forward": (r"blocks\.(\d+)\.cross_attn\.norm_k_img\.weight", r"blocks.\1.attn2.norm_added_k.weight"),
                "backward": (r"blocks\.(\d+)\.attn2\.norm_added_k\.weight", r"blocks.\1.cross_attn.norm_k_img.weight"),
            },
            {"forward": (r"img_emb\.proj\.0\.", r"condition_embedder.image_embedder.norm1."), "backward": (r"condition_embedder\.image_embedder\.norm1\.", r"img_emb.proj.0.")},
            {"forward": (r"img_emb\.proj\.1\.", r"condition_embedder.image_embedder.ff.net.0.proj."), "backward": (r"condition_embedder\.image_embedder\.ff\.net\.0\.proj\.", r"img_emb.proj.1.")},
            {"forward": (r"img_emb\.proj\.3\.", r"condition_embedder.image_embedder.ff.net.2."), "backward": (r"condition_embedder\.image_embedder\.ff\.net\.2\.", r"img_emb.proj.3.")},
            {"forward": (r"img_emb\.proj\.4\.", r"condition_embedder.image_embedder.norm2."), "backward": (r"condition_embedder\.image_embedder\.norm2\.", r"img_emb.proj.4.")},
            {"forward": (r"blocks\.(\d+)\.self_attn\.norm_q\.weight", r"blocks.\1.attn1.norm_q.weight"), "backward": (r"blocks\.(\d+)\.attn1\.norm_q\.weight", r"blocks.\1.self_attn.norm_q.weight")},
            {"forward": (r"blocks\.(\d+)\.self_attn\.norm_k\.weight", r"blocks.\1.attn1.norm_k.weight"), "backward": (r"blocks\.(\d+)\.attn1\.norm_k\.weight", r"blocks.\1.self_attn.norm_k.weight")},
            {"forward": (r"blocks\.(\d+)\.cross_attn\.norm_q\.weight", r"blocks.\1.attn2.norm_q.weight"), "backward": (r"blocks\.(\d+)\.attn2\.norm_q\.weight", r"blocks.\1.cross_attn.norm_q.weight")},
            {"forward": (r"blocks\.(\d+)\.cross_attn\.norm_k\.weight", r"blocks.\1.attn2.norm_k.weight"), "backward": (r"blocks\.(\d+)\.attn2\.norm_k\.weight", r"blocks.\1.cross_attn.norm_k.weight")},
            # head projection mapping
            {"forward": (r"^head\.head\.", "proj_out."), "backward": (r"^proj_out\.", "head.head.")},
        ]

        if direction == "forward":
            return [rule["forward"] for rule in unified_rules]
        elif direction == "backward":
            return [rule["backward"] for rule in unified_rules]
        else:
            raise ValueError(f"Invalid direction: {direction}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def convert_weights(args):
    if os.path.isdir(args.source):
        src_files = glob.glob(os.path.join(args.source, "*.safetensors"), recursive=True)
    elif args.source.endswith((".pth", ".safetensors", "pt")):
        src_files = [args.source]
    else:
        raise ValueError("Invalid input path")

    merged_weights = {}
    logger.info(f"Processing source files: {src_files}")
    for file_path in tqdm(src_files, desc="Loading weights"):
        logger.info(f"Loading weights from: {file_path}")
        if file_path.endswith(".pt") or file_path.endswith(".pth"):
            weights = torch.load(file_path, map_location="cpu", weights_only=True)
        elif file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                weights = {k: f.get_tensor(k) for k in f.keys()}

        duplicate_keys = set(weights.keys()) & set(merged_weights.keys())
        if duplicate_keys:
            raise ValueError(f"Duplicate keys found: {duplicate_keys} in file {file_path}")
        merged_weights.update(weights)

    rules = get_key_mapping_rules(args.direction, args.model_type)
    converted_weights = {}
    logger.info("Converting keys...")
    for key in tqdm(merged_weights.keys(), desc="Converting keys"):
        new_key = key
        for pattern, replacement in rules:
            new_key = re.sub(pattern, replacement, new_key)
        converted_weights[new_key] = merged_weights[key]

    os.makedirs(args.output, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.source))[0] if args.source.endswith((".pth", ".safetensors")) else "converted_model"

    index = {"metadata": {"total_size": 0}, "weight_map": {}}

    chunk_idx = 0
    current_chunk = {}
    for idx, (k, v) in tqdm(enumerate(converted_weights.items()), desc="Saving chunks"):
        current_chunk[k] = v
        if (idx + 1) % args.chunk_size == 0 and args.chunk_size > 0:
            output_filename = f"{base_name}_part{chunk_idx}.safetensors"
            output_path = os.path.join(args.output, output_filename)
            logger.info(f"Saving chunk to: {output_path}")
            st.save_file(current_chunk, output_path)
            for key in current_chunk:
                index["weight_map"][key] = output_filename
            index["metadata"]["total_size"] += os.path.getsize(output_path)
            current_chunk = {}
            chunk_idx += 1

    if current_chunk:
        output_filename = f"{base_name}_part{chunk_idx}.safetensors"
        output_path = os.path.join(args.output, output_filename)
        logger.info(f"Saving final chunk to: {output_path}")
        st.save_file(current_chunk, output_path)
        for key in current_chunk:
            index["weight_map"][key] = output_filename
        index["metadata"]["total_size"] += os.path.getsize(output_path)

    # Save index file
    index_path = os.path.join(args.output, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Index file written to: {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Model weight format converter")
    parser.add_argument("-s", "--source", required=True, help="Input path (file or directory)")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    parser.add_argument("-d", "--direction", choices=["forward", "backward"], default="forward", help="Conversion direction: forward = 'lightx2v' -> 'Diffusers', backward = reverse")
    parser.add_argument("-c", "--chunk-size", type=int, default=100, help="Chunk size for saving (only applies to forward), 0 = no chunking")
    parser.add_argument("-t", "--model_type", choices=["wan"], default="wan", help="Model type")

    args = parser.parse_args()

    if os.path.isfile(args.output):
        raise ValueError("Output path must be a directory, not a file")

    logger.info("Starting model weight conversion...")
    convert_weights(args)
    logger.info(f"Conversion completed! Files saved to: {args.output}")


if __name__ == "__main__":
    main()
