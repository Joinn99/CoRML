import argparse
from CoRML.utils import RecTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='yelp2018', help='Name of Datasets')
    args, _ = parser.parse_known_args()

    RecTrainer(dataset=args.dataset).train(verbose=False)