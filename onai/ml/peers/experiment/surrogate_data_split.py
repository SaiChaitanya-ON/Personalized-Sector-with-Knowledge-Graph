import argparse
import functools
import logging
import os
import shutil
import tempfile
from random import choices

from onai.ml.spark import get_spark
from onai.ml.spark_cache import (
    get_and_add_written_rows,
    get_dbenv,
    get_opened_index,
    setup_logger,
)

INDEX_SUFFIX = ["train", "val", "test"]

logger = logging.getLogger(__name__)


def write_to_disk_cache(row, args):
    # see if it is train, val, or test
    setup_logger()
    index_suffix = choices(
        INDEX_SUFFIX,
        args.split_ratio,
        cum_weights=None,
        k=1,  # I have no bucking clue why this is needed
    )[0]
    idx = get_opened_index(os.path.join(args.o, index_suffix), args)

    idx.append(row)


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("--split_ratio", nargs=3, default=[0.9, 0.1, 0.0], type=float)
    parser.add_argument("--dbenv", default=tempfile.mkdtemp())
    parser.add_argument("-p", default=3)

    args = parser.parse_args()
    spark = get_spark(memory="4g", n_threads=args.p)
    df = spark.read.load(args.i)

    if os.path.exists(args.o):
        raise RuntimeError(f"Output directory {args.o} exists")

    os.mkdir(args.o)
    df.foreach(functools.partial(write_to_disk_cache, args=args))

    spark.stop()
    for i in INDEX_SUFFIX:
        idx = get_opened_index(os.path.join(args.o, i), args)
        idx.close()

    shutil.rmtree(args.dbenv)


if __name__ == "__main__":
    main()
