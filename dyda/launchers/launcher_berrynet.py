import argparse
import json
import logging

from os.path import join as pjoin

import cv2

from dt42lab.core import tools
from dt42lab.utility import dict_comparator
from dyda.components import binary_array_reader
from dyda.components import frame_reader
from dyda.pipelines import pipeline as dt42pl


class BerryNetPipelineLauncher(object):
    def __init__(self, config, dyda_config_path='',
                 output_dirpath='', verbosity=0, lab_flag=False):
        self.pipeline = dt42pl.Pipeline(
            config,
            parent_result_folder=output_dirpath,
            verbosity=verbosity,
            lab_flag=lab_flag)

    def run(self, bitmap, meta={}):
        """Run pipeline.

        Args:
            bitmap: Image data in BGR format (numpy array)

        Returns:
            Dictionary with contents or empty list.
        """
        self.pipeline.run(bitmap, external_meta=meta)
        return self.pipeline.output


def get_args(argv=None):
    """ Prepare auguments for running the script. """

    parser = argparse.ArgumentParser(
        description='Pipeline.'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=(
            '/home/shared/customer_data/acti/201711-ACTi-A/'
            '20171207_recording/acti_2017-12-07-1701/frame'),
        help='Input folder for ')
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='/home/shared/DT42/test_data/'
        'test_auto_labeler_with_tracker/results/',
        help='Output folder for output_metadata')
    parser.add_argument(
        '--lab_flag',
        dest='lab_flag',
        action='store_true',
        default=False,
        help='True to enable related lab process.'
    )
    parser.add_argument(
        '-p', '--pipeline_config',
        type=str,
        default='/home/lab/dt42-dyda/pipeline.config',
        help='File contains the definition of pipeline application.'
    )
    parser.add_argument(
        '-t', '--dyda_config',
        type=str,
        default='',
        help='File contains the component definitions.'
    )
    parser.add_argument(
        "-v", "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity"
    )
    return parser.parse_args(argv)


def main():
    """ Example for testing pipeline. """

    args = get_args()

    logger = logging.getLogger('launcher')
    log_level = logging.WARNING
    if args.verbosity == 1:
        log_level = logging.INFO
    elif args.verbosity >= 2:
        log_level = logging.DEBUG
    formatter = logging.Formatter('[launcher] %(levelname)s %(message)s')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.addHandler(console)

    logger.debug('lab_flag is %r' % args.lab_flag)

    pipeline = BerryNetPipelineLauncher(
        config=args.pipeline_config,
        dyda_config_path=args.dyda_config,
        output_dirpath=args.output,
        verbosity=args.verbosity,
        lab_flag=args.lab_flag
    )

    logger.debug('Running Reader and Selector for frames')
    source_dirpath = args.input
    input_number = 100
    for i in range(input_number):
        input_data = pjoin(source_dirpath, '00000{}.png'.format(570 + i))
        ext_data = cv2.imread(input_data)

        output_data = pipeline.run(ext_data)
        logger.debug('===== frame #{} ====='.format(i))
        logger.debug('input: {}'.format(input_data))
        if (len(output_data) > 0):
            with open('output_{}.json'.format(i), 'w') as f:
                json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
