import argparse

from lightx2v_train.infer import build_inferencer
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained LightX2V model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    model = build_model(config)
    model.load_components()

    inferencer = build_inferencer(config)
    inferencer.set_model(model)
    inferencer.infer()


if __name__ == "__main__":
    main()
