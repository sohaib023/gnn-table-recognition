import argparse
from iterators.table_adjacency_parsing_iterator import TableAdjacencyParsingIterator
from libs.configuration_manager import ConfigurationManager as gconfig


if __name__ != "__main__":
    print("Execute as a python script. This is not an importable module.")
    exit(0)


parser = argparse.ArgumentParser(description='Run training/testing/validation for graph based clustering')
parser.add_argument('input', help="Path to config file")
# parser.add_argument('config', help="Config section within the config file")
parser.add_argument('--test', action="store_true", help="Whether to run inference on test set.")
parser.add_argument('--evaluate', action="store_true", help="Whether to run evaluation on inferenced results.")
# parser.add_argument('--visualize', action="store_true", help="Whether to run layer wise visualization (x-mode only)")
args = parser.parse_args()


gconfig.init(args.input, "conv_graph_dgcnn_fast_conv")

trainer = TableAdjacencyParsingIterator()

if args.test:
    trainer.test()
    if args.evaluate:
    	trainer.evaluate()
# elif args.profile:
#     trainer.profile()
elif args.evaluate:
    trainer.evaluate()
else:
    trainer.train()
