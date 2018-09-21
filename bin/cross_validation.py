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
        description='Orthonormal residuals cross-validation')

    parser.add_argument('-i',
                        '--ti',
                        type=str,
                        metavar='TI_FILENAME',
                        help='Path of training image in gslib format')
    parser.add_argument('-n',
                        '--nsim',
                       type=int,
                       metavar='N_SIMULATIONS',
                       help='Number of random realisations of each point')
    parser.add_argument("-c",
                        "--cdata",
                        type=str,
                        metavar='DATA_FILENAME',
                        required=True,
                        help="Conditioning data input file")

    args = parser.parse_args()

    ti_filename = args.ti
    number_of_simulations = args.nsim
    data_filename = args.cdata
    
    mpstool.cross_validation.cross_validation(ti_filename, data_filename, number_of_simulations)
