import argparse
from src.noise import generate_noisy_dataset
from src.paths import BSD400, BSD68, BSD400_NOISE, BSD68_NOISE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    for src, dst in [(BSD400, BSD400_NOISE), (BSD68, BSD68_NOISE)]:
        generate_noisy_dataset(src=src, dst=dst, seed=args.seed)


if __name__ == "__main__":
    main()