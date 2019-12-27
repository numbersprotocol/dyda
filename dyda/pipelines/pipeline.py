""" Run auto-labeling process for frames. """
import logging
import sys
import time
import os
import copy
import datetime
import importlib
from collections import OrderedDict

from dt42lab.core import tools
from dt42lab.core.pandas_data import is_pandas_df

MODULE_BASE = "dyda.components"


class Pipeline(object):
    """ Main class of the pipeline framework """

    def __init__(self, pipeline_config_path, dyda_config_path="",
                 force_run_skip=False, parent_result_folder="",
                 verbosity=-1, lab_flag=False):
        """ __init__ of TrainerBase """

        self.logger = logging.getLogger('pipeline')
        formatter1 = logging.Formatter(
            '[pipeline] %(levelname)s %(message)s'
        )
        log_level = logging.WARNING
        if verbosity == 1:
            log_level = logging.INFO
        elif verbosity >= 2:
            log_level = logging.DEBUG
        console1 = logging.StreamHandler()
        console1.setFormatter(formatter1)
        self.logger.setLevel(log_level)
        self.logger.addHandler(console1)
        self.log_level = log_level
        self.force_run_skip = force_run_skip

        self.lab_flag = lab_flag
        if parent_result_folder == "":
            cwd = os.getcwd()
            parent_result_folder = os.path.join(cwd, 'results')
        self.logger.info('Saving output to %s' % parent_result_folder)
        tools.check_dir(parent_result_folder)
        self.parent_result_folder = parent_result_folder

        self.pipeline_config = tools.parse_json(pipeline_config_path, 'utf-8')

        if tools.check_exist(dyda_config_path):
            self.dyda_cfg_path = dyda_config_path
        elif "dyda_config" in self.pipeline_config.keys():
            self.dyda_cfg_path = self.pipeline_config["dyda_config"]
        else:
            self.logger.warning(
                "No valid dyda config found by Pipeline, use default."
            )
            self.dyda_cfg_path = ""

        self.pipeline = OrderedDict({})
        self.define_pipeline()
        # Output is only set if output_type is specified by a component
        # If both components set output_type, the later one wins
        self.output = None

        self.trigger_level = "L3"
        if "trigger_level" in self.pipeline_config.keys():
            if self.pipeline_config["trigger_level"] in ["L1", "L2", "L3"]:
                self.trigger_level = self.pipeline_config["trigger_level"]
                self.logger.info(
                    'Changing trigger level to %s' % self.trigger_level
                )

    def get_component_class(self, component, class_name, dyda_cfg=None):
        """ Used for creating the component class instance """

        component_module = MODULE_BASE + '.' + component
        class_instance = getattr(
            importlib.import_module(component_module), class_name
        )
        if dyda_cfg is None:
            class_object = class_instance(
                dyda_config_path=self.dyda_cfg_path
            )
        else:
            class_object = class_instance(
                dyda_config_path=dyda_cfg
            )
        class_object.lab_flag = self.lab_flag
        class_object.reset_logger(self.log_level)
        return class_object

    def define_pipeline(self):
        """ Define pipeline """

        pipeline_def = self.pipeline_config["pipeline_def"]

        # read pipeline from pipeline_def and make an ordered dict
        order = 1
        for comp_config in pipeline_def:
            self.pipeline[comp_config["name"]] = {}
            for key in comp_config:
                if key == "name":
                    continue
                else:
                    self.pipeline[comp_config["name"]][key] = comp_config[key]
            _component = comp_config['component']
            _class = comp_config['class']
            _dyda_cfg = None
            if comp_config['type'] == "skip":
                if self.force_run_skip:
                    self.logger.warning(
                        'force_run_skip is on, skip components will be run'
                    )
                else:
                    continue
            if "dyda_config" in comp_config.keys():
                _dyda_cfg = {_class: comp_config["dyda_config"]}
                self.logger.debug(
                    'Replacing dyda config of %s %s by pipeline session.'
                    % (_class, comp_config["name"])
                )
            self.logger.debug('Initializing %s %s'
                              % (_class, comp_config["name"]))
            _class_instance = self.get_component_class(
                _component, _class, dyda_cfg=_dyda_cfg
            )
            _class_instance.comp_name = comp_config["name"]
            self.pipeline[comp_config["name"]]['instance'] = _class_instance
            self.pipeline[comp_config["name"]]['order'] = order
            self.pipeline[comp_config["name"]]['output'] = None
            order = order + 1

    def run(self, external_data, base_name=None,
            external_meta={}, benchmark=False, force_snapshot=False):
        """ Main function to be called by external code

        @param external_data: data passed by the external function, used by
                              component with data_type == "external"

        """

        if benchmark:
            t0 = time.time()
            # create two strings to compute time of create DataFrame
            t_benchmark_start = []
            t_benchmark_end = []

            t_benchmark_start.append(time.time())
            # create pandas DataFrame to store benchmark
            pd = importlib.import_module('pandas')
            dfcols = ['total time', 'initial setting', 'run',
                      'output_handling']
            benchmark_data = pd.DataFrame(columns=dfcols)
            t_benchmark_end.append(time.time())

        if base_name is not None:
            pass
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            base_name = "snapshot_" + timestamp

        logging.info("Using %s as base_name")

        idx = 0
        previous_comp = ''
        skip_this = False
        skip_following = False

        for name, comp in self.pipeline.items():
            if benchmark:
                t1 = time.time()
            self.logger.debug('running component: %s' % name)

            # if force_run_skip is on, do not skip
            if comp['type'] == 'skip':
                if self.force_run_skip:
                    self.logger.info(
                        'force_run_skip is on, skip components will be run'
                    )
                else:
                    continue

            comp['instance'].reset()

            # if it is the first component
            if idx == 0:
                self.logger.debug('Set base_name as %s' % base_name)
                comp['instance'].metadata[0] = base_name
                comp['instance'].external_metadata = external_meta
                comp['instance'].external_data = external_data

            else:
                comp['instance'].metadata = \
                    self.pipeline[previous_comp]['instance'].metadata
                comp['instance'].external_metadata = external_meta
                comp['instance'].external_data = external_data

            pipeline_status = 0 if skip_following else 1
            comp['instance'].pipeline_status = pipeline_status

            idx = idx + 1
            if skip_this and comp['type'] != 'output_generator':
                self.logger.debug('Skipping the component %s' % name)
                comp['instance'].append_results()
                skip_this = False
                continue

            if skip_following and comp['type'] != 'output_generator':
                self.logger.debug('Skipping the component %s' % name)
                comp['instance'].append_results()
                continue

            input_type = comp['input_type']
            self.logger.debug('Setting up input, type: %s' % input_type)
            if input_type == "use_external_data":
                comp['instance'].input_data = external_data

            elif input_type == "package_external_data":
                comp['instance'].input_data = [external_data]

            elif input_type == "use_external_meta":
                comp['instance'].input_data = external_meta

            elif input_type == "package_external_meta":
                comp['instance'].input_data = [external_meta]

            elif input_type == "append_previous_output":
                comp['instance'].input_data.append(
                    self.pipeline[previous_comp]['instance'].output_data
                )
            elif input_type == "use_metadata":
                comp['instance'].input_data = \
                    copy.deepcopy(comp['instance'].metadata)

            elif input_type == "use_previous_attr":
                attr = comp['additional_info']['attribute']
                if not isinstance(attr, str):
                    self.logger.error(
                        "input_data should be string which match the "
                        "attribute you want to use from previous component"
                    )
                    sys.exit(0)
                comp['instance'].input_data = \
                    getattr(self.pipeline[previous_comp]['instance'], attr)

            elif input_type == "use_previous_output":
                comp['instance'].input_data = \
                    self.pipeline[previous_comp]['instance'].output_data

            elif input_type == "package_previous_output":
                comp['instance'].input_data = \
                    [self.pipeline[previous_comp]['instance'].output_data]

            elif input_type == "append_previous_results":
                comp['instance'].input_data.append(
                    self.pipeline[previous_comp]['instance'].results
                )
            elif input_type == "package_previous_results":
                comp['instance'].input_data = \
                    [self.pipeline[previous_comp]['instance'].results]

            elif input_type == "use_previous_results":
                comp['instance'].input_data = \
                    self.pipeline[previous_comp]['instance'].results

            elif input_type == "use":
                comp['instance'].input_data = external_data
                if len(comp['additional_info']['input_data']) > 1:
                    self.logger.error(
                        "Input type use can only accept one input, "
                        "please check your pipeline config."
                    )
                    sys.exit(0)
                to_use = comp['additional_info']['input_data'][0]
                if to_use[0] == 'external':
                    if to_use[1] == 'data':
                        comp['instance'].input_data = external_data
                    else:
                        comp['instance'].input_data = external_meta
                elif to_use[1] == 'results' or to_use[1] == 'metadata':
                    comp['instance'].input_data = \
                        self.pipeline[to_use[0]]['instance'].results
                elif to_use[1] == 'package_results':
                    comp['instance'].input_data = \
                        [self.pipeline[to_use[0]]['instance'].results]
                elif to_use[1] == 'output_data':
                    comp['instance'].input_data = \
                        self.pipeline[to_use[0]]['instance'].output_data
                else:
                    self.logger.error(
                        '%s %s to_use type is not supported'
                        % (to_use[0], to_use[1])
                    )

            elif input_type == "append":
                if 'additional_info' not in comp.keys():
                    self.logger.error('No additional_info found')
                    sys.exit(0)
                if 'input_data' not in comp['additional_info'].keys():
                    self.logger.error('No input_data session'
                                      'found in additional_info')
                    sys.exit(0)
                for to_append in comp['additional_info']['input_data']:
                    if to_append[0] == 'external':
                        if to_append[1] == 'data':
                            comp['instance'].input_data.append(external_data)
                        else:
                            comp['instance'].input_data.append(external_meta)
                    elif to_append[1] == 'metadata':
                        comp['instance'].append_metadata_to_input(
                            self.pipeline[to_append[0]]['order'],
                            self.pipeline[to_append[0]]['class']
                        )
                    elif to_append[1] == 'results':
                        comp['instance'].input_data.append(
                            self.pipeline[to_append[0]]['instance'].results
                        )
                    elif to_append[1] == 'package_results':
                        comp['instance'].input_data.append(
                            [self.pipeline[to_append[0]]['instance'].results]
                        )
                    elif to_append[1] == 'output_data':
                        comp['instance'].input_data.append(
                            self.pipeline[to_append[0]]['instance'].output_data
                        )
                    else:
                        self.logger.error(
                            '%s %s to_append type is not supported'
                            % (to_append[0], to_append[1])
                        )
            elif input_type == "extend":
                if 'additional_info' not in comp.keys():
                    self.logger.error('No additional_info found')
                    sys.exit(0)
                if 'input_data' not in comp['additional_info'].keys():
                    self.logger.error('No input_data session'
                                      'found in additional_info')
                    sys.exit(0)
                for to_extend in comp['additional_info']['input_data']:
                    if to_extend[0] == 'external':
                        if to_extend[1] == 'data':
                            comp['instance'].input_data.extend(external_data)
                        else:
                            comp['instance'].input_data.extend(external_meta)
                    elif to_extend[1] == 'metadata':
                        comp['instance'].extend_input_with_meta(
                            self.pipeline[to_extend[0]]['order'],
                            self.pipeline[to_extend[0]]['class']
                        )
                    elif to_extend[1] == 'package_results':
                        comp['instance'].input_data.extend(
                            [self.pipeline[to_append[0]]['instance'].results]
                        )
                    elif to_extend[1] == 'results':
                        comp['instance'].input_data.extend(
                            self.pipeline[to_extend[0]]['instance'].results
                        )
                    elif to_extend[1] == 'output_data':
                        comp['instance'].input_data.extend(
                            self.pipeline[to_extend[0]]['instance'].output_data
                        )
                    else:
                        self.logger.error(
                            '%s %s to_extend type is not supported'
                            % (to_extend[0], to_extend[1])
                        )
                        sys.exit(0)

            elif input_type == "use_meta_pairs":
                if 'additional_info' not in comp.keys():
                    self.logger.error('No additional_info found')
                comp['instance'].input_data = \
                    comp['additional_info']['comp_key_pairs']

            else:
                self.logger.error('No valid input_type found.')

            if benchmark:
                t2 = time.time()
            snapshot_folder = os.path.join(
                self.parent_result_folder, comp['instance'].class_name
            )
            comp['instance'].snapshot_folder = snapshot_folder
            comp['instance'].run()
            if comp['instance'].terminate_flag:
                self.logger.warning(
                    'Component reports terminate_flag, the following normal '
                    'components will be skipped. Pipeline will go straight '
                    'to output_generator.'
                )
                self.output = None
                skip_following = True
                continue

            if benchmark:
                t3 = time.time()

            # selector behavior: if it is a selector, pass the next one if fail
            if comp['type'] == 'selector':
                # if the data does not pass selector, e.g. not a key frame
                if not comp['instance'].output_data:
                    skip_this = True

            # gate behavior: if it is a gate, pass all following but output
            elif comp['type'] == 'gate':
                # if the data does not pass gate, e.g. not a key frame
                if not comp['instance'].output_data:
                    skip_following = True

            if 'base_name' in comp.keys():
                if isinstance(comp["base_name"], str):
                    comp['instance'].metadata[0] = comp["base_name"]
                    self.logger.warning(
                        "Changing base_name as %s" % comp["base_name"]
                    )
                else:
                    self.logger.error("Fail to change base_name")

            self.logger.debug('metadata: {}'.format(comp['instance'].metadata))

            # make snaoshot if output_type is specified
            if 'output_type' in comp.keys():
                if self.lab_flag is True:
                    self.logger.info('Passing output step because '
                                     'lab_flag is set as True')
                    pass
                else:
                    do_snapshot = False
                    '''
                    snapshot == True
                        => always create snapshot
                    force_snapshotable == True
                        => snapshot when force_snapshot is True
                    '''
                    if 'snapshot' in comp.keys() and comp['snapshot'] is True:
                        do_snapshot = True
                    elif ('force_snapshotable' in comp.keys() and
                          comp['force_snapshotable'] is True):
                        if force_snapshot:
                            do_snapshot = True
                    if comp['output_type'] == 'metadata':
                        self.output = comp['instance'].metadata
                        if do_snapshot:
                            comp['instance'].snapshot_metadata()
                            self.logger.debug('Snapshot for metadata of %s'
                                              % comp['instance'].metadata[0])
                    elif comp['output_type'] == 'output_data':
                        self.output = comp['instance'].output_data
                        if do_snapshot:
                            if False not in [is_pandas_df(i)
                                             for i in self.output]:
                                comp['instance'].snapshot_output_data(
                                    dtype='DataFrame'
                                )
                            else:
                                comp['instance'].snapshot_output_data()
                            self.logger.debug('Making snapshot for %s'
                                              % comp['instance'].metadata[0])
                    elif comp['output_type'] == 'unpack_results':
                        if len(comp['instance'].results) >= 1:
                            comp['instance'].results = \
                                comp['instance'].results[0]
                            self.output = comp['instance'].results
                        else:
                            self.logger.error(
                                "Unpacking results fail, "
                                "keep component results unchanged"
                            )
                        if do_snapshot:
                            comp['instance'].snapshot_results()
                            self.logger.debug('Snapshot for results of %s'
                                              % comp['instance'].metadata[0])
                    elif comp['output_type'] == 'results':
                        self.output = comp['instance'].results
                        if do_snapshot:
                            comp['instance'].snapshot_results()
                            self.logger.debug('Snapshot for results of %s'
                                              % comp['instance'].metadata[0])
                    elif comp['output_type'] == 'post_process':
                        self.output = [
                            comp['instance'].results,
                            comp['instance'].metadata,
                            comp['instance'].output_data,
                        ]
                        self.logger.info('Running post_process under'
                                         'lab_flag = %r' % self.lab_flag)
                        comp['instance'].post_process(
                            out_folder_base=self.parent_result_folder
                        )
                    else:
                        self.logger.warning('%s output_type is not supported'
                                            % comp['output_type'])
                        pass

                comp['output'] = self.output

            if 'print_output' in comp.keys() and comp['print_output'] is True:
                print('[pipeline] printing Pipeline.output of %s (%s)'
                      % (comp['instance'].comp_name,
                         comp['instance'].class_name))
                print(self.output)

            if benchmark:
                t4 = time.time()
                print('[pipeline] Total time used for %s is %.5f seconds'
                      % (comp['instance'].class_name, t4 - t1))
                print(' --- Time used for initial setting is %.5f seconds'
                      % (t2 - t1))
                print(' --- Time used for component run is %.5f seconds'
                      % (t3 - t2))
                print(' --- Time used for output handling is %.5f seconds'
                      % (t4 - t3))

                t_benchmark_start.append(time.time())
                component_benchmark = pd.Series(
                    data=[t4 - t1, t2 - t1,
                          t3 - t2, t4 - t3],
                    index=dfcols)
                benchmark_data.loc[comp['instance'].class_name, :] = \
                    component_benchmark
                t_benchmark_end.append(time.time())

            previous_comp = name

        if benchmark:
            pipeline_time = ((time.time() - t0) -
                             (sum(t_benchmark_end) - sum(t_benchmark_start)))
            print('Total time used for pipeline.run() is %.5f seconds'
                  % pipeline_time)
            benchmark_data.ix['Total time used for pipeline.run()', 0] = \
                pipeline_time
            benchmark_data[:] = benchmark_data[:].apply(pd.to_numeric)
            benchmark_folder = os.path.join(self.parent_result_folder,
                                            'benchmark')
            tools.check_dir(benchmark_folder)
            benchmark_path = os.path.join(benchmark_folder, base_name + '.csv')
            benchmark_data.to_csv(path_or_buf=benchmark_path,
                                  float_format='%.5f')
        return True
