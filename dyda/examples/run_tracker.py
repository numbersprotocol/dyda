""" Run tracker for frames. """
import logging
import argparse
import os
import importlib

from dyda_utils import data
from dyda_utils import tools
from dyda.components import label_image


def set_default_values(values_list_file):
    """ Set default_values for dyda components """

    default_values = {
        'tracker': 'TrackerByOverlapRatio'
    }

    if tools.check_exist(values_list_file, log=False):
        try:
            default_values_ = data.parse_json(values_list_file)
            for key in default_values.keys():
                if key not in default_values_.keys():
                    default_values_[key] = default_values[key]
            default_values = default_values_
            logging.info('Read default values from %s' % values_list_file)
        except Exception:
            logging.warning('Fail to read %s' % values_list_file)

    else:
        logging.info('No file found, use default settings.')
    return default_values


def get_args(argv=None):
    """ Prepare auguments for running the script """

    parser = argparse.ArgumentParser(
        description='Trainer.'
    )
    parser.add_argument(
        '-f', '--file_of_values', type=str, default="",
        help='File contains the default values.'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='/home/shared/DT42/test_data/test_tracker/results/',
        help='Output folder for detection results added track id')
    parser.add_argument(
        '-d',
        '--data_list',
        type=str,
        default='/home/shared/DT42/test_data/test_tracker/json_list.txt',
        help="List of file paths with detection results in json")
    parser.add_argument(
        "-v", "--verbosity", action="count", default=1,
        help="increase output verbosity"
    )
    return parser.parse_args(argv)


def main():
    """ Example of how to use tracker

    """

    args = get_args()

    log_level = logging.WARNING
    if args.verbosity == 1:
        log_level = logging.INFO
    elif args.verbosity == 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level,
                        format='[Labeler %(levelname)s] %(message)s')

    data_list = args.data_list
    labler_values = set_default_values(args.file_of_values)
    result_folder = args.output
    tools.check_dir(result_folder)

    logging.info('Initializing Tracker.')
    tracker_class = getattr(
        importlib.import_module("dyda.components.tracker"),
        labler_values['tracker']
    )
    tracker_ = tracker_class()

    fpaths = data.txt_to_list(data_list)
    for json_filename in fpaths:
        print('process: ' + json_filename)
        tracker_.reset_metadata()
        tracker_.input_data = data.parse_json(json_filename)
        tracker_.run()
        if not tracker_.results == []:
            output_filename = os.path.join(
                result_folder,
                tools.replace_extension(
                    tracker_.results['filename'],
                    'json'))
            data.write_json(
                tracker_.results,
                output_filename)


if __name__ == "__main__":
    main()
