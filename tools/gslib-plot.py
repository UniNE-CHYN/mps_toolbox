#!/usr/bin/env python3
import argparse
import numpy as np
import mpstool


if __name__ == '__main__':
    # Description of the program after using -h
    parser = argparse.ArgumentParser(
        description='Visualise GSLIB image')

    # There is only one positional argument
    parser.add_argument('filename',
                        type=str,
                        metavar='filename',
                        help='Path of file to be visualised')
    # Parse and save local variables
    args = parser.parse_args()
    filename = args.filename

    mpstool.img.Image.fromGslib(filename).plot()
