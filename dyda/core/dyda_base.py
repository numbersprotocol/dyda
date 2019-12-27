import os
import sys
import datetime
import traceback
import logging
import copy
import numpy as np

from dyda_utils import tools
from dyda_utils import data
from dyda_utils import image
from dyda_utils import lab_tools


class TrainerBase(object):

    def __init__(self, dyda_config_path='', log_level=logging.INFO):
        """ __init__ of TrainerBase

        TrainerBase.input_data
            A list of dictionaries, list or Numpy.arrays,
            the values should be given in run.
        TrainerBase.external_data
            Data assigned by pipeline or external function
        TrainerBase.external_metadata
            Metadata assigned by pipeline or external function
        TrainerBase.metadata
            A list to include the results or metadata created and used
            by every component. The results from each component will be
            attached to the TrainerBase.metadata after run is executed.
            The first element of the list is base_name of the component, and
            the default value is timestamp unless user changes it or
            read_data of DataReaderBase is executed.
            The output_data will be saved as $base_name.jpg.0,
            $base_name.jpg.1, etc, and the metadata will be saved as
            $base_name.json in when snapshot is executed.
        TrainerBase.comp_name
            Name given by pipeline
        TrainerBase.config
            Global configuration which can be accessed by all child classes.
            The content will be read from /etc/dyda/dyda.config by
            default, and it will remains as an empty dictionary if
            dyda.config cannot be foune. One can also call
            set_dyda_config to reset.
        TrainerBase.param
            This should be the configuration used by child classes.
            It remains empty until set_param is called.
        TrainerBase.output_data
            The output_data is a list which stores the processed data
            of TrainerBase.input_data in main_process.
        TrainerBase.results
            The results after executing each component. This will be added
            to metadata after run is executed.
        TrainerBase.terminate_flag
            If severe error occurs, one should set terminate_flag as True.
            Pipeline will skip the following components and reset the flag.
        TrainerBase.lab_flag
            Used for determining if this run is for production or lab internal.
            Default will be False
        TrainerBase.lab_output_folder
            This variable should ONLY be accessed by pre_process and
            post_process. The purpose is for assigning the parent folder
            of output which is only used by lab members.
        TrainerBase.snapshot_folder
            This variable is assigned when snapshop is triggered.
        TrainerBase.snapshot_fnames
            Keep the snapshot file names to be accessible by external module
            Format: [$SNAPSHOT_PATH1, $SNAPSHOT_PATH2]

        :Example:

        Example test

        .. seealso:: foo
        .. warning:: foo
        .. note:: foo
        .. todo:: foo

        """

        self.class_name = self.__class__.__name__
        self.comp_name = ""
        self.base_logger = logging.getLogger('dyda_base_' + self.class_name)
        self.logger = logging.getLogger(self.class_name)
        self.reset_logger(log_level)
        self.input_data = []
        self.external_metadata = {}
        self.external_data = None
        self.metadata = ['']
        self._check_base_name()
        self.config = {}
        self.param = {}
        self.output_data = []
        self.init_dyda_config(dyda_config_path, terminate_if_fail=False)
        self.results = {'additional_info': {}}
        self.lab_flag = False
        self.terminate_flag = False
        self.pipeline_status = 1
        self.lab_output_folder = ""
        self.snapshot_folder = ""
        self.snapshot_fnames = []
        self.package = False

        self.set_param(self.class_name)

        self.snapshot_hierarchy = False
        # see dyda issues 122 for more details
        if "snapshot_hierarchy" in self.param.keys():
            self.snapshot_hierarchy = self.param["snapshot_hierarchy"]

        # if force_snapshotable is True, component will snapshot if
        # force_snapshot is set as True in pipeline and launcher
        self.force_snapshotable = False
        if "force_snapshotable" in self.param.keys():
            self.force_snapshotable = self.param["force_snapshotable"]

        # if unpack_single_list is True, dyda will unpack the list
        # which has only one component (i.e. [a] => a)
        self.unpack_single_list = True
        if "unpack_single_list" in self.param.keys():
            self.unpack_single_list = self.param["unpack_single_list"]

        # if snapshot_with_counter is True, dyda will add counter to
        # the end of the filename while doing snapshot (i.e. $FNAME_0.jpg)
        self.snapshot_with_counter = True
        if "snapshot_with_counter" in self.param.keys():
            self.snapshot_with_counter = self.param["snapshot_with_counter"]
            if self.snapshot_with_counter is False:
                self.logger.warning(
                    "snapshot_with_counter is on, no counter value is added"
                    "to the end of snapshot images"
                )

    def reset_logger(self, log_level):
        """ reset logger """

        formatter1 = logging.Formatter(
            '[dyda_base] %(levelname)s %(message)s'
        )
        console1 = logging.StreamHandler()
        console1.setFormatter(formatter1)
        self.base_logger.setLevel(log_level)
        self.base_logger.addHandler(console1)

        formatter2 = logging.Formatter(
            '[' + self.class_name + '] %(levelname)s %(message)s',
        )
        console2 = logging.StreamHandler()
        console2.setFormatter(formatter2)
        self.logger.setLevel(log_level)
        self.logger.addHandler(console2)

    def reset_metadata(self):
        """ Reset metadata but keep the first element untouched."""

        self.logger.debug("Reset metadata for %s" % self.class_name)
        self.metadata = self.metadata[:1]

    def reset(self):
        """ Reset metadata and reset input_data to empty list."""

        self.base_logger.debug("Reset everything for %s" % self.class_name)
        self.terminate_flag = False
        self.pipeline_status = 1
        self.reset_metadata()
        self.reset_input_data()
        self.snapshot_fnames = []
        self.reset_output_data()
        self.reset_results()

    def reset_output_data(self):
        """ Reset output_data, this should be defined in the base component."""
        pass

    def reset_input_data(self):
        """ Reset input_data """
        self.input_data = []

    def reset_results(self):
        """ Reset results, this should be defined in the base component. """
        pass

    def reset_output(self):
        """ Reset output_data and results"""

        self.reset_output_data()
        self.reset_results()

    def append_metadata_to_input(self, session_order, class_name_to_check):
        """ This function append the specific session of TrainerBase.metadata
            to TrainerBase.input_data and check if the class name is the
            expected one

        :param int session_order: the order of the session in metadata, please
                                  note, since the first (index 0) element of
                                  metadata is always base_name, the order
                                  of the component session starts from 1

        :param str class_name_to_check: the class name to be checked
        :return: None
        :rtype: None
        """

        self.base_logger.debug("Append metadata to input_data of %s"
                               % self.class_name)
        try:
            session_meta = copy.deepcopy(self.metadata[session_order])
        except IndexError:
            self.base_logger.error('Cannot find the specified session_order '
                                   'in metadata, it does not exist, '
                                   'please check.')
            self.terminate_flag = True
        _class_name = session_meta['class_name']
        if _class_name != class_name_to_check:
            self.base_logger.error('The value of class_name does not match, '
                                   'session %s vs %s'
                                   % (_class_name, class_name_to_check))
            self.terminate_flag = True
        else:
            self.input_data.append(session_meta['results'])

    def extend_input_with_meta(self, session_order, class_name_to_check):
        """ This function extend the TrainerBase.input_data with the specific
            session of TrainerBase.metadata and check if the class
            name is the expected one

        :param int session_order: the order of the session in metadata, please
                                  note, since the first (index 0) element of
                                  metadata is always base_name, the order
                                  of the component session starts from 1

        :param str class_name_to_check: the class name to be checked
        :return: None
        :rtype: None
        """

        self.base_logger.debug("Extend input_data of %s with metadata"
                               % self.class_name)
        try:
            session_meta = copy.deepcopy(self.metadata[session_order])
        except IndexError:
            self.base_logger.error('Cannot find the specified session_order '
                                   'in metadata, it does not exist, '
                                   'please check.')
            self.terminate_flag = True
        _class_name = session_meta['class_name']
        if _class_name != class_name_to_check:
            self.base_logger.error('The value of class_name does not match, '
                                   'session %s vs %s'
                                   % (_class_name, class_name_to_check))
            self.terminate_flag = True
        else:
            self.input_data.extend(session_meta['results'])

    def init_dyda_config(self, dyda_config, encoding=None,
                            terminate_if_fail=True):
        """ Initialize TrainerBase.config

        :param str dyda_config: Input for setting up config.It can be a
                                   read dictionary or a path of the config json
        :param str encoding: Choose encoding of the config json file
        :param bool terminate_if_fail: True to terminate the programe if the
                                       setting of TrainerBase.config fails.
        :return: None
        :rtype: None
        """

        self.base_logger.info("Initializing dyda.config using "
                              "/etc/dyda/dyda.config")
        default_path = '/etc/dyda/dyda.config'
        self.set_dyda_config(
            default_path, encoding=encoding,
            terminate_if_fail=terminate_if_fail
        )
        self.base_logger.info("Reading dyda.config in $HOME/.dyda/"
                              "dyda.config for session %s "
                              % self.class_name)
        home_path = os.path.join(os.path.expanduser('~'),
                                 '.dyda/dyda.config')
        self.replace_dyda_config(
            home_path, self.class_name, encoding=encoding,
            terminate_if_fail=terminate_if_fail
        )
        self.base_logger.info("Trying to read config session for %s "
                              % self.class_name)
        self.replace_dyda_config(
            dyda_config, self.class_name, encoding=encoding,
            terminate_if_fail=terminate_if_fail
        )

    def set_dyda_config(self, dyda_config, encoding=None,
                           terminate_if_fail=True):
        """ Set TrainerBase.config

        :param str dyda_config: Input for setting up config.It can be a
                                   read dictionary or a path of the config json
        :param str encoding: Choose encoding of the config json file
        :param bool terminate_if_fail: True to terminate the programe if the
                                       setting of TrainerBase.config fails.
        :return: None
        :rtype: None
        """

        if isinstance(dyda_config, dict):
            self.config = dyda_config
            self.base_logger.info('TrainerBase.config set successfully')

        elif os.path.isfile(dyda_config):
            try:
                self.config = data.parse_json(
                    dyda_config, encoding=encoding
                )
            except IOError:
                self.base_logger.error('Cannot open %s' % dyda_config)
                if terminate_if_fail:
                    sys.exit(0)
            except Exception:
                traceback.print_exc(file=sys.stdout)
                if terminate_if_fail:
                    sys.exit(0)

    def replace_dyda_config(self, dyda_config, session,
                               encoding=None, terminate_if_fail=True):
        """ Replace the whole TrainerBase.config or just partially

        :param str dyda_config: Input for setting up config.It can be a
                                   read dictionary or a path of the config json
        :param str session: Specify the component session, all for all sessions
        :param str encoding: Choose encoding of the config json file
        :param bool terminate_if_fail: True to terminate the programe if the
                                       setting of TrainerBase.config fails.
        :return: None
        :rtype: None
        """

        if session == 'all':
            self.set_dyda_config(
                dyda_config, encoding=encoding,
                terminate_if_fail=terminate_if_fail
            )
        else:
            config = None

            if isinstance(dyda_config, dict):
                config = dyda_config

            elif os.path.isfile(dyda_config):
                try:
                    config = data.parse_json(
                        dyda_config, encoding=encoding
                    )
                except IOError:
                    self.base_logger.error('Cannot open %s' % dyda_config)
                    if terminate_if_fail:
                        sys.exit(0)
                except Exception:
                    traceback.print_exc(file=sys.stdout)
                    if terminate_if_fail:
                        sys.exit(0)
            if config:
                if session in config.keys():
                    self.config[session] = config[session]
                else:
                    if isinstance(dyda_config, dict):
                        self.base_logger.warning('%s session not found in '
                                                 'input dyda_config dict'
                                                 % session)
                    elif isinstance(dyda_config, str):
                        self.base_logger.warning('%s session not found in %s'
                                                 % (session, dyda_config))
                    else:
                        self.base_logger.warning('%s session not found'
                                                 % session)

    def set_param(self, session, param=None):
        """
        set_param will reset self.param according to the TrainerBase.config
        If TrainerBase.config is not yet set, it uses the given values

        :param session: TBD
        :param dict param: Parameters used by the selector. The config
                           will be used with higher priority.
        :return: None
        :rtype: None
        """

        self.base_logger.debug("Running set_param of %s" % self.class_name)
        if param is not None:
            self.param = param
            self.base_logger.info('Reset param using given values for '
                                  '%s.' % session)
        if session in self.config.keys():
            self.param = self.config[session]
            self.base_logger.warning('Reset param using %s in config'
                                     % session)
        else:
            self.base_logger.warning('Fail to apply, param of session %s '
                                     'remains unchanged.' % session)

    def _check_base_name(self):
        """
        Check base_name, return timestamp as string if it is None or empty
        """

        if self.metadata[0] == "" or self.metadata[0] is None:
            self.metadata[0] = tools.create_timestamp()
        else:
            pass

    def check_snapshot_folder(self):
        """ Check if self.snapshot_folder is properly set """

        if self.snapshot_folder == "":
            self.snapshot_folder = os.getcwd()
            self.snapshot_folder = os.path.join(
                self.snapshot_folder, self.class_name
            )
            self.base_logger.info(
                "Creating snapshot folder at %s." % self.snapshot_folder
            )
        tools.check_dir(self.snapshot_folder)

    def create_hierarchy_outf(self, snapshot_type):
        """
        create hierarchy output folder based on self.snapshot_folder

        snapshot_type: type of output to be snapshot (results, metadata, etc)
        """

        now = datetime.datetime.now()
        m = str(now.month)
        d = str(now.day)
        h = str(now.hour)
        out_folder = os.path.join(
            self.snapshot_folder, snapshot_type, m, d, h
        )
        return out_folder

    def snapshot(self):
        """
        snapshot function will output a json file for
        TrainerBase.metadata. It will also save a copy of the
        current TrainerBase.data based on the data_type.

        :return: None
        :rtype: None
        """

        self.base_logger.debug("Creating snapshot for %s" % self.class_name)
        self.snapshot_metadata(out_name=self.metadata[0])
        self.snapshot_output_data(out_name=self.metadata[0])
        self.snapshot_results(out_name=self.metadata[0])

    def snapshot_results(self, out_name=''):
        """
        snapshot_results will save a copy of the current
        TrainerBase.results as json output

        :param str out_name: preffix of the output files (default: snapshot)
        :return: None
        :rtype: None
        """

        self.base_logger.debug("Creating snapshot of results for %s"
                               % self.class_name)
        self.check_snapshot_folder()

        if self.snapshot_hierarchy:
            out_folder = self.create_hierarchy_outf('results')
        else:
            out_folder = os.path.join(self.snapshot_folder, 'results')

        if out_name == "":
            if isinstance(self.metadata[0], str):
                out_name = self.metadata[0]
            else:
                out_name = "snapshot"
        try:
            tools.check_dir(out_folder)
        except Exception:
            self.base_logger.error('Cannot create folder %s' % out_folder)
            traceback.print_exc(file=sys.stdout)
        try:
            if isinstance(self.results, list):
                # snapshot results in only one json
                if "snapshot_results_all" in self.param.keys() \
                        and self.param["snapshot_results_all"] is True:
                    results_path = os.path.join(
                        out_folder, out_name + '.json'
                    )
                    tools.write_json(self.results, fname=results_path)
                    return True
                # snapshot list of results as different json
                for i in range(0, len(self.results)):
                    tmp_path = os.path.join(
                        out_folder, out_name + '_tmp.json'
                    )
                    if self.snapshot_with_counter:
                        results_path = os.path.join(
                            out_folder, out_name + '_' + str(i) + '.json'
                        )
                    else:
                        results_path = os.path.join(
                            out_folder, out_name + '.json'
                        )
                    tools.write_json(self.results[i], fname=tmp_path)
                    os.rename(tmp_path, results_path)
            else:
                results_path = os.path.join(
                    out_folder, out_name + '.json'
                )
                tools.write_json(self.results, fname=results_path)
        except Exception:
            self.base_logger.error('Cannot make snapshot for metadata.')
            traceback.print_exc(file=sys.stdout)

        return True

    def snapshot_metadata(self, out_name=''):
        """
        snapshot_metadata will save a copy of the current
        TrainerBase.metadata as json output

        :param str out_name: preffix of the output files (default: snapshot)
        :return: None
        :rtype: None
        """

        self.base_logger.debug("Creating snapshot of meta for %s"
                               % self.class_name)
        self.check_snapshot_folder()

        if self.snapshot_hierarchy:
            out_folder = self.create_hierarchy_outf('metadata')
        else:
            out_folder = os.path.join(self.snapshot_folder, 'metadata')

        if out_name == "":
            if isinstance(self.metadata[0], str):
                out_name = self.metadata[0]
            else:
                out_name = "snapshot"
        try:
            tools.check_dir(out_folder)
        except Exception:
            self.base_logger.error('Cannot create folder %s' % out_folder)
            traceback.print_exc(file=sys.stdout)
        try:
            metadata_path = os.path.join(out_folder, out_name + '.json')
            data.write_json(self.metadata, fname=metadata_path)
            self.snapshot_fnames.append(metadata_path)
        except Exception:
            self.base_logger.error('Cannot make snapshot for metadata.')
            traceback.print_exc(file=sys.stdout)

    def snapshot_output_data(self, dtype="image", out_name=''):
        """
        snapshot_data will save a copy of the current TrainerBase.data
        based on the data_type set.

        :param str dtype: data type of output data
        :param str out_name: preffix of the output files (default: snapshot)
        :return: None
        :rtype: None
        """

        self.base_logger.info('Snaoshot output data type: %s.' % dtype)
        self.check_snapshot_folder()

        if self.snapshot_hierarchy:
            out_folder = self.create_hierarchy_outf('output_data')
        else:
            out_folder = os.path.join(self.snapshot_folder, 'output_data')

        if out_name == "":
            if isinstance(self.metadata[0], str):
                out_name = self.metadata[0]
            else:
                out_name = "snapshot"
        try:
            tools.check_dir(out_folder)
        except Exception:
            self.base_logger.error('Cannot create folder' % out_folder)
            traceback.print_exc(file=sys.stdout)
        try:
            counter = 0
            if not isinstance(self.output_data, list):
                output = [self.output_data]
            else:
                output = self.output_data
            for output_data_ in output:
                if dtype == 'image':
                    _tmp = os.path.join(
                        out_folder, out_name + '_tmp.jpg'
                    )
                    if self.snapshot_with_counter:
                        output_img_path = os.path.join(
                            out_folder, out_name + '_' + str(counter) + '.jpg'
                        )
                    else:
                        output_img_path = os.path.join(
                            out_folder, out_name + '.jpg'
                        )
                    image.save_img(output_data_, fname=_tmp)
                    os.rename(_tmp, output_img_path)
                    self.snapshot_fnames.append(output_img_path)
                elif dtype == 'DataFrame':
                    _tmp = os.path.join(
                        out_folder, out_name + '_tmp.csv'
                    )
                    if self.snapshot_with_counter:
                        output_img_path = os.path.join(
                            out_folder, out_name + '_' + str(counter) + '.csv'
                        )
                    else:
                        output_img_path = os.path.join(
                            out_folder, out_name + '.csv'
                        )
                    output_data_.to_csv(_tmp)
                    os.rename(_tmp, output_img_path)
                    self.snapshot_fnames.append(output_img_path)
                else:
                    self.base_logger.error('Type %s is not yet supported'
                                           % dtype)
                counter = counter + 1
        except Exception:
            self.base_logger.error('Fail to snapshot for output_data'
                                   'and metadata')
            traceback.print_exc(file=sys.stdout)

    def pre_process(self):
        """
        Pre-processing is called before the main_process.
        """
        pass

    def main_process(self):
        """
        main_process function will be called in the run function after
        pre_process and before post_process. The main logical computation
        of data should happen here.
        """
        pass

    def post_process(self, out_folder_base=""):
        """
        post_process function will be called in the run function after
        main_process.

        Warning: sample code below may be overwritten by components

        Arguments:
            out_folder_base - parent output folder of post_process results
        """
        if not tools.check_exist(out_folder_base, log=False):
            out_folder_base = os.path.join(os.getcwd(), 'post_process')
        self.base_logger.info('post_process results saved to %s'
                              % out_folder_base)
        tools.dir_check(out_folder_base)

    def run(self):
        """
        run should be called by the external function, and it calls
        pre_process, main_process and post_process in orders.

        """

        if self.lab_flag is True:
            self.pre_process()
            self.main_process()
            self.append_results()
            self.post_process()
        else:
            self.main_process()
            self.append_results()

        return self.metadata

    def append_results(self):
        """
        Append the TrainerBase.results and TrainerBase.class_name
        as the key.
        """

        self.base_logger.debug("Append results of %s to metadata."
                               % self.class_name)
        meta_to_append = {'class_name': self.class_name,
                          'comp_name': self.comp_name,
                          'results': self.results}
        self.metadata.append(meta_to_append)

    def check_meta_length(self, include_comp=None, use_comp_name=False,
                          exclude_key=["external"]):
        """
        Check if length of each meta session is consistent

        Arguments:
            include_comp: list of component names, None to check all
        """
        length = -1
        if use_comp_name:
            meta_key = "comp_name"
        else:
            meta_key = "class_name"

        for meta_sess in self.metadata[1:]:
            if include_comp is not None:
                if meta_sess[meta_key] not in include_comp:
                    continue
            if meta_sess[meta_key] in exclude_key:
                continue

            results = meta_sess["results"]

            if isinstance(results, list):
                _length = len(results)
            else:
                _length = 1
            if length < 0:
                length = _length
            else:
                if length != _length:
                    return False
        return length

    def unpack_list(self, input_list):
        """ Unpack input list if the length is one """

        if isinstance(input_list, list):
            if len(input_list) == 1:
                input_data = copy.deepcopy(input_list)
                return input_data[0]
            else:
                self.logger.debug(
                    "Input list size > 1, unpack_single_list does nothing."
                )
                return input_list
        else:
            self.logger.debug(
                "Input is not list, unpack_single_list does nothing."
            )
            return input_list

    def unpack_single_output(self):
        if isinstance(self.output_data, list):
            if len(self.output_data) == 1:
                self.output_data = self.output_data[0]

    def unpack_single_input(self):
        if isinstance(self.input_data, list):
            if len(self.input_data) == 1:
                self.input_data = self.input_data[0]

    def unpack_single_results(self):
        if isinstance(self.results, list) and len(self.results) == 1:
            self.results = self.results[0]

    def pack_input_as_list(self):
        """ Check if input_data is a list, package it if not """
        self.input_data = self.pack_as_list(self.input_data)

    def pack_as_list(self, item_to_check):
        """ Check if the item is a list, package it if not """

        if isinstance(item_to_check, list):
            return item_to_check
        else:
            return [item_to_check]

    def compare_single_inputs(self, input_a, input_b):
        """
        Compare two inputs, if one is dict and another one is a list with only
        one component, package the dict as a list as well
        """
        if isinstance(input_a, dict):
            if isinstance(input_b, list):
                if len(input_b) == 1:
                    return [input_a], input_b
        if isinstance(input_b, dict):
            if isinstance(input_a, list):
                if len(input_a) == 1:
                    return input_a, [input_b]
        return input_a, input_b

    def check_param_keys(self, default_keys):
        """
        Check if any default key is missing in the self.param.
        :param list default_keys: a list of keys to be checked.
        """

        for _key in default_keys:
            if _key not in self.param.keys():
                self.base_logger.error('%s missing in self.param' % _key)
                sys.exit(0)
            else:
                continue
        self.base_logger.info('keys of self.param are checked')

    def uniform_input(self, _input_data=None, dtype=None,
                      terminate_if_not_valid=True):
        """ Package input_data if it is not a list and
            check input data type
        """

        # package input_data if it is not a list
        if _input_data is None:
            input_data = copy.deepcopy(self.input_data)
        else:
            input_data = copy.deepcopy(_input_data)
        if isinstance(input_data, list):
            self.package = False
        else:
            input_data = [input_data]
            self.package = True

        # check input data type and
        valid = True
        for data in input_data:
            if dtype == "str":
                if not isinstance(data, str):
                    valid = False
            elif dtype == "lab-format":
                if not lab_tools.if_result_match_lab_format(data):
                    valid = False
            elif dtype == "ndarray":
                if not isinstance(data, np.ndarray):
                    valid = False
            else:
                self.base_logger.info('dtype is not checked')

        # when data type is not valid, raise terminate_flag and
        # return empty list to skip following computation
        if not valid:
            self.base_logger.error('Invalid input data type')
            if terminate_if_not_valid:
                self.terminate_flag = True

        return input_data

    def uniform_output(self):
        """ Un-package output_data and results if they are packaged before.

        """
        if self.package:
            self.unpack_single_results()
            self.unpack_single_output()
