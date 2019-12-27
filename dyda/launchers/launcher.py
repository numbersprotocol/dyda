import os
import sys
import argparse
import logging

from dyda_utils import tools
from dyda_utils import dict_comparator
from dyda.components import frame_reader
from dyda.pipelines import pipeline as dt42pl


def get_args(argv=None):
    """ Prepare auguments for running the script """

    parser = argparse.ArgumentParser(
        description='Pipeline.'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='',
        help='Output folder for output_metadata')
    parser.add_argument(
        '-j',
        '--json_list',
        type=str, default=None,
        help="Path to the text file of json lists."
    )
    parser.add_argument(
        '-d',
        '--data_list',
        type=str,
        default='/home/shared/DT42/test_data/'
        'test_auto_labeler_with_tracker/frame_list.txt',
        help="Path to the text file of train images list"
    )
    parser.add_argument(
        '--direct_input', dest='direct_input',
        action='store_true', default=False,
        help="True to use data_list as direct input (i.e. no txt to list)"
    )
    parser.add_argument(
        '-m',
        '--repeated_metadata_path',
        type=str,
        default='',
        help="Path of external metadata feed in pipeline repeatedly."
    )
    parser.add_argument(
        '--lab_flag', dest='lab_flag', action='store_true', default=False,
        help='True to enable related lab process.'
    )
    parser.add_argument(
        '--benchmark', dest='benchmark', action='store_true', default=False,
        help='True to enable pipeline benchmark of time used.'
    )
    parser.add_argument(
        '--force_run_skip', dest='force_run_skip',
        action='store_true', default=False,
        help='True to force running skip components.'
    )
    parser.add_argument(
        '--read_frame', dest='read_frame', action='store_true', default=False,
        help='True to read frame before sending input to dyda pipeline.'
    )
    parser.add_argument(
        '-p', '--pipeline_config', type=str,
        help='File contains the definition of pipeline application.',
        default='/home/lab/dt42-dyda/pipeline.config'
    )
    parser.add_argument(
        '--dyda_config_path', type=str,
        help='File of dyda config',
        default=''
    )
    parser.add_argument(
        '--read_meta', dest='read_meta', action='store_true', default=False,
        help='Read metadata',
    )
    parser.add_argument(
        "-v", "--verbosity", action="count", default=0,
        help="increase output verbosity"
    )
    parser.add_argument(
        "--loop_over_input", dest='loop_over_input',
        action='store_true', default=False,
        help="Loop over input data_list"
    )
    parser.add_argument(
        "--force_snapshot", dest='force_snapshot',
        action='store_true', default=False,
        help="Force snapshot components with force_snapshotable on."
    )
    parser.add_argument(
        "--do_not_pack", dest='do_not_pack',
        action='store_true', default=False,
        help="True to send data to pipeline directly without packaging as list"
    )
    parser.add_argument(
        "--multi_channels", dest='multi_channels',
        action='store_true', default=False,
        help="True to input 4 channels to pipeline"
    )
    parser.add_argument(
        "--check_output", dest='check_output',
        action='store_true', default=False,
        help="Check output with reference output list given"
    )
    parser.add_argument(
        '-r',
        '--ref_output',
        type=str,
        default='',
        help="Path to the json file with list of reference output"
    )
    parser.add_argument(
        '-ig',
        '--ignore_key',
        type=str,
        default='',
        help=(
            "Ignore key(s) while checking output."
            "Use , to seperate if you have multiple keys"
        )
    )

    return parser.parse_args(argv)


def main():
    """ main function to run pipeline """

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

    logger.info('lab_flag is %r' % args.lab_flag)
    pipeline = dt42pl.Pipeline(
        args.pipeline_config,
        dyda_config_path=args.dyda_config_path,
        parent_result_folder=args.output,
        verbosity=args.verbosity,
        lab_flag=args.lab_flag,
        force_run_skip=args.force_run_skip
    )

    if args.read_frame:
        fr = frame_reader.FrameReader()
    # looping over input data paths
    logger.info('Running Reader and Selector for frames')
    data_list = args.data_list
    if args.json_list:
        logger.warning('json_list will replace -d/--data_list argument')
        data_list = args.json_list

    force_snapshot = False
    if args.force_snapshot:
        force_snapshot = True

    bfile_list = False
    if args.direct_input:
        fpaths = data_list
    elif os.path.isfile(data_list):
        if tools.check_ext(data_list, ".json"):
            fpaths = tools.parse_json(data_list, 'utf-8')
        else:
            fpaths = tools.txt_to_list(data_list)
        bfile_list = True
    elif os.path.isdir(data_list):
        fpaths = []
        bfile_list = False
    else:
        logger.error("Something wrong with data_list input, please check")
        sys.exit(0)

    ignore_keys = []
    if len(args.ignore_key) > 1:
        ignore_keys = args.ignore_key.split(',')
    all_pass = False
    if args.check_output:
        logger.debug(args.ref_output)
        if os.path.isdir(args.ref_output):
            fn_list = sorted(tools.find_files(
                dir_path=args.ref_output, keyword=None,
                suffix=('.json'), walkin=True))
            ref_output = []
            for fn in fn_list:
                ref_output.append(tools.parse_json(fn, 'utf-8'))
        elif os.path.isfile(args.ref_output):
            ref_output = tools.parse_json(args.ref_output, 'utf-8')
        else:
            logger.error("Something wrong with reference output, please check")
            sys.exit(0)
        all_pass = True

    benchmark = False
    if args.benchmark:
        benchmark = True

    if bfile_list and args.loop_over_input and args.multi_channels:
        for fi in range(len(fpaths)):
            ext_data = []
            ext_meta = []
            for ci in range(len(fpaths[fi])):
                full_path = fpaths[fi][ci]
                logger.debug(full_path)
                if args.read_frame:
                    logger.debug('Reading frame for producing binary input')
                    fr.reset()
                    fr.input_data = [full_path]
                    fr.run()
                    ext_data.append(fr.output_data[0])
                else:
                    ext_data.append(full_path)
                ext_meta.append(fpaths[fi][ci])
            ext_meta = read_meta_single(args, logger, full_path)

            pipeline.run(
                ext_data, external_meta=ext_meta, benchmark=benchmark,
                force_snapshot=force_snapshot
            )
            if args.check_output:
                if not isinstance(pipeline.output, list):
                    tar_list = [pipeline.output]
                    ref_list = [ref_output[fi]]
                else:
                    tar_list = pipeline.output
                    ref_list = ref_output[fi]
                for ci, tar_data in enumerate(tar_list):
                    all_pass = check_result(
                        tar_data, ref_list[ci],
                        full_path, all_pass, ignore_keys=ignore_keys)

    elif bfile_list and args.loop_over_input:
        counter = 0
        wrong = 0
        for fi in range(len(fpaths)):

            counter = counter + 1
            full_path = fpaths[fi]
            logger.debug(full_path)
            base_name = tools.remove_extension(full_path,
                                               return_type='base-only')
            # Assign external data and metadata
            if args.do_not_pack:
                ext_data = full_path
            else:
                ext_data = [full_path]
            if args.read_frame:
                logger.debug('Reading frame for producing binary input')
                fr.reset()
                fr.input_data = [full_path]
                fr.run()
                ext_data = fr.output_data[0]
            ext_meta = read_meta_single(args, logger, full_path)
            pipeline.run(
                ext_data, base_name=base_name,
                external_meta=ext_meta, benchmark=benchmark,
                force_snapshot=force_snapshot
            )

            if args.check_output:
                all_pass = check_result(
                    pipeline.output,
                    ref_output[fi],
                    full_path,
                    all_pass,
                    ignore_keys=ignore_keys)
    elif bfile_list:
        ext_meta = []
        if args.read_meta:
            logger.debug('Reading json for producing binary meta')
            for full_path in fpaths:
                if args.repeated_metadata_path == '':
                    meta_path = tools.remove_extension(full_path) + '.json'
                else:
                    meta_path = args.repeated_metadata_path
                try:
                    ext_meta.append(tools.parse_json(meta_path, 'utf-8'))
                except BaseException:
                    logger.error('Fail to parse %s' % meta_path)
                    sys.exit(0)

        pipeline.run(
            fpaths, external_meta=ext_meta, benchmark=benchmark,
            force_snapshot=force_snapshot
        )

        if args.check_output:
            all_pass = check_result(
                pipeline.output,
                ref_output,
                fpaths,
                all_pass,
                ignore_keys=ignore_keys)

    else:
        full_path = data_list
        ext_meta = read_meta_single(args, logger, full_path)
        pipeline.run(
            full_path, external_meta=ext_meta, benchmark=benchmark,
            force_snapshot=force_snapshot
        )
        if args.check_output:
            all_pass = check_result(
                pipeline.output[0],
                ref_output[0],
                fpaths,
                all_pass,
                ignore_keys=ignore_keys)

    if args.check_output is True and all_pass is True:
        print("Pass all test data in input data list.")
    print("Lab pipeline launcher completes successfully")


def check_result(tar, ref, input_filename, all_pass, ignore_keys=[]):
    if tar == [] and ref == []:
        pass
    else:
        report = dict_comparator.get_diff(ref, tar, ignore_keys=ignore_keys)
        if report['extra_field'] == [] and \
                report['missing_field'] == []and \
                report['mismatch_val'] == []:
            pass
        else:
            print('[Launcher] Output is different with reference: %s'
                  % input_filename)
            all_pass = False
    return all_pass


def read_meta_single(args, logger, full_path):
    ext_meta = []
    if args.read_meta:
        logger.debug('Reading json for producing binary meta')
        if args.repeated_metadata_path == '':
            meta_path = tools.remove_extension(full_path) + '.json'
        else:
            meta_path = args.repeated_metadata_path
        try:
            ext_meta = tools.parse_json(meta_path, 'utf-8')
        except BaseException:
            logger.error('Fail to parse %s' % meta_path)
            sys.exit(0)
    return ext_meta


if __name__ == "__main__":
    main()
