import argparse
from bank_agent.data import prepare
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/generated")
    p.add_argument("--train", type=int, default=500)
    p.add_argument("--eval", type=int, default=200)
    args = p.parse_args()
    print(prepare(args.output, args.train, args.eval))
