#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint

import numpy

from groundhog.trainer.SGD_adadelta import SGD as SGD_adadelta
from groundhog.mainLoop import MainLoop
from nmt import RNNEncoderDecoder, prototype_state, 
from helper import get_batch_iterator
import nmt

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--proto",  default="prototype_state",
        help="Prototype state to use for state")
    parser.add_argument("--skip-init", action="store_true",
        help="Skip parameter initilization")
    parser.add_argument("changes",  nargs="*", help="Changes to state", default="")
    return parser.parse_args()

def main():
    args = parse_args()

    state = getattr(nmt, args.proto)()
    if args.state:
        if args.state.endswith(".py"):
            state.update(eval(open(args.state).read()))
        else:
            with open(args.state) as src:
                state.update(cPickle.load(src))
    for change in args.changes:
        state.update(eval("dict({})".format(change)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug("State:\n{}".format(pprint.pformat(state)))

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, args.skip_init)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()

    logger.debug("Load data")
    train_data = get_batch_iterator(state)
    logger.debug("Compile trainer")
    algo = eval(state['algo'])(lm_model, state, train_data)
    logger.debug("Run training")
    main = MainLoop(train_data, None, None, lm_model, algo, state, None,
            reset=state['reset'],
            hooks=[RandomSamplePrinter(state, lm_model, train_data)]
                if state['hookFreq'] >= 0
                else None)
    if state['reload']:
        main.load()
    if state['loopIters'] > 0:
        main.main()

if __name__ == "__main__":
    main()
