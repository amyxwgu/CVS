import argparse
import logging

logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

def SetLogging(logfile):
    lfh = logging.FileHandler(logfile, mode='w')
    lch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    lfh.setFormatter(formatter)
    lch.setFormatter(formatter)
    logger.addHandler(lfh)
    logger.addHandler(lch)
    return

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Python code for continuous non-monotone submodular function optimization for query-based video summarization")

    parser.add_argument('-d', '--dataset', type=str, required=True, help="path to h5 dataset (required)")
    parser.add_argument('-s', '--split', type=str, required=True, help="path to split file (required)")
    parser.add_argument('--split-id', type=int, default=0, help="split index (default: 0)")
    parser.add_argument('--max-iter', type=int, default=5, help="maximum iteration for training (default: 5)")
    parser.add_argument('--eval-iter', type=int, default=100, help="maximum iteration for evaluation (default: 100)")
    parser.add_argument('--lr', type=float, default=1e-05, help="learning rate (default: 1e-05)")
    parser.add_argument('--eval-lr', type=float, default=3e-02, help="learning rate (default: 3e-02)")
    parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
    parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
    parser.add_argument('--save-dir', type=str, default='QVSmodels/',
                        help="path to save output (default: 'QVSmodels/')")
    parser.add_argument('--verbose', action='store_false', help="whether to show detailed test results")
    parser.add_argument('--save-results', action='store_false', help="whether to save output results")
    parser.add_argument('-m', '--metric', type=str, required=True, choices=['OVP', 'Youtube'],
                        help="evaluation metric ['OVP', 'Youtube']")
    parser.add_argument('--mbsize', type=int, default=1, help="batch size for train & test (default: 1)")
    parser.add_argument('--epoch', type=int, default=2, help="training epoch (default: 2)")
    parser.add_argument('--optim', type=str, default='AdaGrad', help="SGD / SGDM / AdaGrad / RMSProp")
    parser.add_argument('--ftype', type=str, default='googlenet', help="feature type: googlenet / color")
    parser.add_argument('--cond', action='store_true', help="whether to use conditional video summarization")
    parser.add_argument('--mode', type=int, default=1,
                        help="1: train deep submodularity / 2: evaluation only")
    parser.add_argument('--query', type=int, default=0, help="query frame index")

    return parser


def get_arguments() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args
