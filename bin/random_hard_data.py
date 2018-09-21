#!/usr/bin/env python3

import argparse
import mpstool

def sample_random_data_from_file(filename, n_samples):
    image = mpstool.img.Image.fromGslib(filename).asArray()
    return mpstool.cross_validation.sample_random_conditioning_data(image[:,:,0], n_samples)

def sample_random_data_from_file_window(filename, n_samples, window):
    image = mpstool.img.Image.fromGslib(filename).get_sample((window[0], window[1], 0)).asArray()
    return mpstool.cross_validation.sample_random_conditioning_data(image[:,:,0], n_samples)

def generate_and_write_random_conditioning_data(ti_filename, n_samples, output_filename):
    data = sample_random_data_from_file(ti_filename, n_samples)
    mpstool.cross_validation.save_to_gslib(output_filename, data)

def generate_and_write_random_conditioning_data_window(ti_filename, n_samples, output_filename, window):
    data = sample_random_data_from_file_window(ti_filename, n_samples, window)
    mpstool.cross_validation.save_to_gslib(output_filename, data)


if __name__ == '__main__':
    # Description of the program after using -h
    parser = argparse.ArgumentParser(
        description='Generate random samples from a training image')

    parser.add_argument('-i',
                        '--ti',
                        type=str,
                        metavar='TI_FILENAME',
                        help='Path of training image in gslib format')
    parser.add_argument('-n',
                        '--npoints',
                       type=int,
                       metavar='N_SAMPLES',
                       help='Number of random samples')
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        metavar='OUTPUT_FILENAME',
                        required=True,
                        help="Name of the gslib output file")

    parser.add_argument("-x",
                        "--window_x",
                        type=int,
                        metavar='WINDOW_X_SIZE',
                        required=False,
                        help="Size of the window in x")

    parser.add_argument("-y",
                        "--window_y",
                        type=int,
                        metavar='WINDOW_Y_SIZE',
                        required=False,
                        help="Size of the window in y")

    args = parser.parse_args()

    ti_filename = args.ti
    number_of_samples = args.npoints
    output_filename = args.output
    
    if args.window_x or args.window_y is not None:
        if args.window_x and args.window_y is None:
            parser.error("You must specify none or both -x and -y flags.")
        else:
            window = (args.window_x, args.window_y)
            generate_and_write_random_conditioning_data_window(ti_filename, number_of_samples, output_filename, window)

    else:
        generate_and_write_random_conditioning_data(ti_filename, number_of_samples, output_filename)


