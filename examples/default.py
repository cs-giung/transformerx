"""It provides a default setup shared across all example scripts."""
import sys
sys.path.append('./') # pylint: disable=wrong-import-position

import os
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from datetime import datetime
from typing import Callable, Tuple

import jax
from tabulate import tabulate


def str2bool(v): # pylint: disable=missing-function-docstring
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise ArgumentTypeError(f'boolean value expected, got {v}')


def get_args(
        parser: ArgumentParser,
        exist_ok: bool = False,
        dot_log_file: bool = False,
        libraries: Tuple = (),
    ) -> Tuple[Namespace, Callable]:
    """
    Args:
        parser: 
        exist_ok (bool): it determines whether to permit execution of the
            script when `args.save` already exists or not (default: False).
        dot_log_file (bool): write logs as a dotfile (default: False).
        libraries: a tuple of libraries to be recorded.

    Returns:
        a tuple of parsed namespace and print function.
    """
    parser.add_argument(
        '--save', default=None, type=str,
        help='save outputs in the specified path (default: None)')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='random seed for the script (default: None)')
    args = parser.parse_args()

    if args.seed is None:
        args.seed = (
            os.getpid()
            + int(datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big'))

    if args.save is not None:
        if not exist_ok and os.path.exists(args.save):
            raise AssertionError(
                f'args.save={args.save} already exists. Consider using '
                f'`exist_ok=True` to permit duplicate executions.')
        os.makedirs(args.save, exist_ok=True)

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    def print_fn(s):
        s = datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + s
        if args.save is not None:
            if exist_ok:
                out = f'console.{time_stamp}.log'
                if dot_log_file:
                    out = '.' + out
            else:
                out = 'console.log'
            out = os.path.join(args.save, out)
            with open(out, 'a', encoding='utf-8') as fp:
                fp.write(s + '\n')
        print(s, flush=True)

    log_str = tabulate([
        ('sys.platform', sys.platform),
        ('python', sys.version.replace('\n', ''))] + [(
            lib.__name__,
            lib.__version__ + ' @' + os.path.dirname(lib.__file__)
        ) for lib in libraries])
    log_str = f'Environments:\n{log_str}\n'
    print_fn(log_str)

    log_str = ''
    max_k_len = max(map(len, vars(args).keys()))
    for k, v in vars(args).items():
        log_str += f'- args.{k.ljust(max_k_len)} : {v}\n'
    log_str = f'Command line arguments:\n{log_str}'
    print_fn(log_str)

    log_str = f'Local devices:\n{jax.local_devices()}\n'
    print_fn(log_str)

    return args, print_fn
