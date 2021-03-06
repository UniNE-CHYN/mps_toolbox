#! /bin/env python3

import geone_cv
import shutil
import pandas as pd
import sys
sys.path.append('../tools/')


def main():
    # run cv as from command line
    geone_cv.run_cv('cv_data/xv_input.json')

    # read generated results by the script
    results = pd.read_csv('results/cv_results.csv')

    # read reference data
    results_ref = pd.read_csv('cv_data/cross-validation-ref.csv')

    # compare relevant features
    features = ['param_TI', 'param_nneighboringNode',
                'mean_test_brier', 'mean_test_zero_one',
                'mean_test_linear', 'mean_test_skill_brier',
                'mean_test_skill_zero_one']
    assert(results[features].equals(results_ref[features]))

    # print message if successful
    print("geone_cv regression test passed")


if __name__ == '__main__':
    try:
        main()
    except AssertionError:
        # results are different
        sys.exit("geone_cv regression test failed")
    finally:
        # clean up output of the script
        shutil.rmtree('results', ignore_errors=True)
