import os
import copy
import numpy as np
from dyda.core import tracker_base
from dt42lab.core import lab_tools
from dt42lab.core import tools


class TrackerByColor(tracker_base.TrackerBase):
    """ Tracking by color cue.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(TrackerByColor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.parse_param()

    def parse_param(self):
        """ Parse parameters used in dyda component. """

        self.max_missing_frame = self.param['max_missing_frame']
        self.counter = 0
        self.objects = []

        if 'type' not in self.param.keys():
            self.type = ['centroid']
        elif isinstance(self.param['type'], list):
            self.type = self.param['type']
        else:
            self.type = [self.param['type']]
        if 'adaptive_thre' not in self.param.keys():
            self.adaptive = -1
        else:
            self.adaptive = self.param['adaptive_thre']
        self.rotate_input = False
        if 'rotate_input' in self.param.keys():
            self.rotate_input = self.param['rotate_input']
        if 'momentum' not in self.param.keys():
            self.momentum = 0
        else:
            self.momentum = self.param['momentum']
        if 'hist_bin_num' not in self.param.keys():
            self.bin_num = 4
        else:
            self.bin_num = self.param['hist_bin_num']
        if 'channel_wise_track_id' not in self.param.keys():
            self.chan_wise = False
        else:
            self.chan_wise = self.param['channel_wise_track_id']

        self.track_id = None
        self.pre_results = None

        self.large_number = float(10**10)
        self.small_number = 1 / self.large_number
        self.thre = self.param['matching_thre']
        if 'centroid' in self.type and 'color' not in self.type:
            self.thre = np.reciprocal(self.thre + self.small_number)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input(self.input_data[0], 'lab-format')
        imgs = self.uniform_input(self.input_data[1], 'list')

        self.rotation = -1
        if self.rotate_input:
            self.rotation = len(input_data)

        if self.track_id is None:
            if self.chan_wise and self.rotate_input:
                self.track_id = [0 for i in range(self.rotation)]
            else:
                self.track_id = 0
        if self.pre_results is None and self.rotate_input:
            self.pre_results = [{} for i in range(self.rotation)]

        if len(self.input_data) == 3:
            self.counter = self.input_data[2]['channel_index']

        if len(self.objects) == 0:
            self.objects = [[] for i in range(len(input_data))]
        elif len(self.objects) != len(input_data):
            self.terminate_flag = True
            self.logger.error("Length of input_data sould be consistent.")

        for i, data in enumerate(input_data):
            if self.rotation > 0 and i != self.counter:
                if self.pre_results[i] == {}:
                    data = self.assign_default_track_id(data)
                else:
                    data = copy.deepcopy(self.pre_results[i])
                self.results.append(data)
                continue
            for j, anno in enumerate(data['annotations']):
                anno['cropped_img'] = imgs[i][j]
            data['annotations'], self.objects[i] = self.update(
                data['annotations'], self.objects[i], i)
            for anno in data['annotations']:
                anno.pop('cropped_img')
            self.results.append(data)

            if self.pre_results is not None:
                self.pre_results[i] = copy.deepcopy(data)

        self.uniform_output()
        if self.rotation > 0:
            self.counter += 1
            if self.counter == self.rotation:
                self.counter = 0

    def assign_default_track_id(self, data):
        """ Assign default track_id as -1. """

        for obj in data['annotations']:
            if 'track_id' not in obj.keys():
                obj['track_id'] = -1
        return data

    def predict_by_momentum(self, now, pre):
        """ Predict new location according to momentum. """

        mx = (now['right'] + now['left']) / 2 - \
            (pre['right'] + pre['left']) / 2
        my = (now['bottom'] + now['top']) / 2 - \
            (pre['bottom'] + pre['top']) / 2
        now['top'] += int(my * self.momentum)
        now['bottom'] += int(my * self.momentum)
        now['left'] += int(mx * self.momentum)
        now['right'] += int(mx * self.momentum)
        return now

    def register(self, obj, objs_track, ch):
        """ Register new objects. """

        if self.chan_wise and self.rotation > 0:
            obj['track_id'] = self.track_id[ch]
            self.track_id[ch] += 1
        else:
            obj['track_id'] = self.track_id
            self.track_id += 1
        objs_track.append(copy.deepcopy(obj))
        objs_track[-1]['missing_frame'] = 0
        return (obj, objs_track)

    def update(self, objs, objs_track, ch):
        """ Update objects tracked. """

        if len(objs) == 0:
            # no object detected and mark tracking objects missing
            for i in range(len(objs_track) - 1, -1, -1):
                objs_track[i]['missing_frame'] += 1

                # deregister tracking objects if they reached a maximum
                # missing frame
                if objs_track[i]['missing_frame'] > self.max_missing_frame:
                    objs_track.pop(i)

            return (objs, objs_track)

        # register all if no tracking object
        if len(objs_track) == 0:
            for obj in objs:
                obj, objs_track = self.register(obj, objs_track, ch)

        # otherwise, match current objects and tracking objects
        else:
            # calculate location similarity matrix D,
            # the bigger value the more similar
            if 'overlap' in self.type:
                D = lab_tools.calculate_overlap_ratio_all(
                    objs, objs_track)
            elif 'IoU' in self.type:
                D = lab_tools.calculate_IoU_all(
                    objs, objs_track)
            elif 'centroid' in self.type:
                D = lab_tools.calculate_centroid_dist_all(
                    objs_track, objs)
                D = np.reciprocal(D + self.small_number)
            else:
                self.logger.warning("Not supported type, use default "
                                    "type instead.")
                self.type = ['centroid']
                D = lab_tools.calculate_centroid_dist_all(
                    objs_track, objs)
                D = np.reciprocal(D + self.small_number)

            # calculate color similarity matrix D,
            # the bigger value the more similar
            if 'color' in self.type:
                D_color = lab_tools.calculate_color_similarity_all(
                    objs_track, objs, self.bin_num)

            # score -1 when labels are different or similarity < self.thre
            skip = np.zeros((len(objs_track), len(objs)), dtype=bool)

            for idx_1, obj_1 in enumerate(objs_track):
                for idx_2, obj_2 in enumerate(objs):
                    if self.adaptive > 0:
                        base_1 = min(
                            obj_1['right'] - obj_1['left'],
                            obj_1['bottom'] - obj_1['top'])
                        base_2 = min(
                            obj_2['right'] - obj_2['left'],
                            obj_2['bottom'] - obj_2['top'])
                        self.adp_thre = min(base_1, base_2) * self.adaptive
                        self.adp_thre = np.reciprocal(
                            self.adp_thre + self.small_number)
                    else:
                        self.adp_thre = self.thre
                    if obj_1['label'] == obj_2['label'] and \
                            D[idx_1, idx_2] > self.adp_thre:
                        continue
                    skip[idx_1, idx_2] = True
                    if 'color' in self.type:
                        D_color[idx_1, idx_2] = -1
                    else:
                        D[idx_1, idx_2] = -1

            # one-to-one match
            if 'color' in self.type:
                D_base = D_color
            else:
                D_base = D

            rows = []
            cols = []
            while D_base.max() > self.thre:
                index = np.argwhere(D_base == D_base.max())
                i = index[0, 0]
                j = index[0, 1]
                rows.append(i)
                cols.append(j)
                D_base[i, :] = -1
                D_base[:, j] = -1

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if skip[row, col]:
                    continue
                # ignore if used
                if row in used_rows or col in used_cols:
                    continue
                # update object and reset missing frame number when matched
                objs[col]['track_id'] = objs_track[row]['track_id']
                if self.momentum > 0 and objs_track[row]['missing_frame'] == 0:
                    objs_track[row] = self.predict_by_momentum(
                        copy.deepcopy(objs[col]),
                        objs_track[row])
                else:
                    objs_track[row] = copy.deepcopy(objs[col])
                objs_track[row]['missing_frame'] = 0

                # indicate that the row and column used
                used_rows.add(row)
                used_cols.add(col)

            # extract unused row and column
            unused_rows = list(set(range(0, D.shape[0])).difference(used_rows))
            unused_cols = list(set(range(0, D.shape[1])).difference(used_cols))

            # update missing objects
            # loop over the unused row
            for r in range(len(unused_rows) - 1, -1, -1):
                row = unused_rows[r]

                # update missing frame number
                objs_track[row]['missing_frame'] += 1

                # deregister tracking objects if they reached a maximum
                # missing frame
                if objs_track[row]['missing_frame'] > self.max_missing_frame:
                    objs_track.pop(row)

            # register new tracking object
            for col in unused_cols:
                obj, objs_track = self.register(objs[col], objs_track, ch)

        # return updated tracking objects
        return (objs, objs_track)

    def uniform_input(self, input_data, dtype):
        """ Package input_data if it is not a list and
            check input data type
        """

        # package input_data if it is not a list
        input_data = copy.deepcopy(input_data)
        if not isinstance(input_data, list):
            input_data = [input_data]
            self.package = True
        else:
            self.package = False

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
            elif dtype == "list":
                if not isinstance(data, list):
                    valid = False
            else:
                self.base_logger.warning('dtype is not supported to check')

        # when data type is not valid, raise terminate_flag and
        # return empty list to skip following computation
        if not valid:
            self.base_logger.error('Invalid input data type')
            self.terminate_flag = True
            input_data = []

        return input_data


class TrackerSimple(tracker_base.TrackerBase):
    """ Tracking by location cue.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(TrackerSimple, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.parse_param()

    def parse_param(self):
        """ Parse parameters used in dyda component. """

        self.max_missing_frame = self.param['max_missing_frame']
        self.track_id = 0
        self.counter = 0
        self.objects = []

        if 'type' not in self.param.keys():
            self.type = 'centroid'
        else:
            self.type = self.param['type']
        if 'adaptive_thre' not in self.param.keys():
            self.adaptive = -1
            self.thre = self.param['matching_thre']
        else:
            self.adaptive = self.param['adaptive_thre']
        if 'momentum' not in self.param.keys():
            self.momentum = 0
        else:
            self.momentum = self.param['momentum']

        self.large_number = float(10**10)
        self.small_number = 1 / self.large_number

        if 'centroid' == self.type and self.adaptive == -1:
            self.thre = np.reciprocal(self.thre + self.small_number)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        if len(self.objects) == 0:
            self.objects = [[] for i in range(len(input_data))]
        elif len(self.objects) != len(input_data):
            self.terminate_flag = True
            self.logger.error("Length of input_data sould be consistent.")

        for i, data in enumerate(input_data):
            data['annotations'], self.objects[i] = self.update(
                data['annotations'], self.objects[i])
            self.results.append(data)

        self.uniform_output()

    def register(self, obj, objs_track):
        """ Register new objects. """

        obj['track_id'] = self.track_id
        objs_track.append(copy.deepcopy(obj))
        objs_track[-1]['missing_frame'] = 0
        self.track_id += 1
        return (obj, objs_track)

    def predict_by_momentum(self, now, pre):
        """ Predict new location according to momentum. """

        mx = (now['right'] + now['left']) / 2 - \
            (pre['right'] + pre['left']) / 2
        my = (now['bottom'] + now['top']) / 2 - \
            (pre['bottom'] + pre['top']) / 2
        now['top'] += int(my * self.momentum)
        now['bottom'] += int(my * self.momentum)
        now['left'] += int(mx * self.momentum)
        now['right'] += int(mx * self.momentum)
        return now

    def update(self, objs, objs_track):
        """ Update objects tracked. """

        if len(objs) == 0:
            # no object detected and mark tracking objects missing
            for i in range(len(objs_track) - 1, -1, -1):
                objs_track[i]['missing_frame'] += 1

                # deregister tracking objects if they reached a maximum
                # missing frame
                if objs_track[i]['missing_frame'] > self.max_missing_frame:
                    objs_track.pop(i)

            return (objs, objs_track)

        # register all if no tracking object
        if len(objs_track) == 0:
            for obj in objs:
                obj, objs_track = self.register(obj, objs_track)

        # otherwise, match current objects and tracking objects
        else:
            if self.type == 'overlap':
                D = lab_tools.calculate_overlap_ratio_all(
                    objs, objs_track)
            elif self.type == 'IoU':
                D = lab_tools.calculate_IoU_all(
                    objs, objs_track)
            elif self.type == 'centroid':
                D = lab_tools.calculate_centroid_dist_all(
                    objs_track, objs)
                D = np.reciprocal(D + self.small_number)
            else:
                self.logger.warning("Not supported type, use default "
                                    "type instead.")
                self.type = ['centroid']
                D = lab_tools.calculate_centroid_dist_all(
                    objs_track, objs)
                D = np.reciprocal(D + self.small_number)

            # score -1 when labels are different or overlap_ratio < th
            skip = np.zeros((len(objs_track), len(objs)), dtype=bool)
            for idx_1, obj_1 in enumerate(objs_track):
                for idx_2, obj_2 in enumerate(objs):
                    if self.adaptive > 0:
                        base_1 = min(
                            obj_1['right'] - obj_1['left'],
                            obj_1['bottom'] - obj_1['top'])
                        base_2 = min(
                            obj_2['right'] - obj_2['left'],
                            obj_2['bottom'] - obj_2['top'])
                        self.thre = min(base_1, base_2) * self.adaptive
                        self.thre = np.reciprocal(
                            self.thre + self.small_number)
                    if obj_1['label'] == obj_2['label'] and \
                            D[idx_1, idx_2] > self.thre:
                        continue
                    skip[idx_1, idx_2] = True
                    D[idx_1, idx_2] = -1

            # one-to-one match
            rows = []
            cols = []
            while D.max() > 0:
                index = np.argwhere(D == D.max())
                i = index[0, 0]
                j = index[0, 1]
                rows.append(i)
                cols.append(j)
                D[i, :] = -1
                D[:, j] = -1

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if skip[row, col]:
                    continue
                # ignore if used
                if row in used_rows or col in used_cols:
                    continue

                # update object and reset missing frame number when matched
                objs[col]['track_id'] = objs_track[row]['track_id']
                if self.momentum > 0 and objs_track[row]['missing_frame'] == 0:
                    objs_track[row] = self.predict_by_momentum(
                        copy.deepcopy(objs[col]),
                        objs_track[row])
                else:
                    objs_track[row] = copy.deepcopy(objs[col])
                objs_track[row]['missing_frame'] = 0

                # indicate that the row and column used
                used_rows.add(row)
                used_cols.add(col)

            # extract unused row and column
            unused_rows = list(set(range(0, D.shape[0])).difference(used_rows))
            unused_cols = list(set(range(0, D.shape[1])).difference(used_cols))

            # update missing objects
            # loop over the unused row
            for r in range(len(unused_rows) - 1, -1, -1):
                row = unused_rows[r]

                # FIXME: unused_rows may not in index of objs_track
                if row >= len(objs_track):
                    continue

                # update missing frame number
                objs_track[row]['missing_frame'] += 1

                # deregister tracking objects if they reached a maximum
                # missing frame
                if objs_track[row]['missing_frame'] > self.max_missing_frame:
                    objs_track.pop(row)

            # register new tracking object
            for col in unused_cols:
                obj, objs_track = self.register(objs[col], objs_track)

        # return updated tracking objects
        return (objs, objs_track)


class TrackerByOverlapRatio(tracker_base.TrackerBase):

    def __init__(self, dyda_config_path="", param=None):
        """ Initialization function of dyda component. """

        super(TrackerByOverlapRatio, self).__init__(
            dyda_config_path=dyda_config_path
        )

        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        if not isinstance(self.param['target'], list):
            self.param_target = [self.param['target']]
        else:
            self.param_target = self.param['target']
        if 'overlap_ratio_th' not in self.param.keys():
            self.param_overlap_ratio_th = {}
            for target in self.param_target:
                self.param_overlap_ratio_th[target] = 0
        else:
            self.param_overlap_ratio_th = self.param['overlap_ratio_th']
        if not isinstance(self.param_overlap_ratio_th, dict):
            th = copy.deepcopy(self.param_overlap_ratio_th)
            self.param_overlap_ratio_th = {}
            for target in self.param_target:
                self.param_overlap_ratio_th[target] = th

        previous_num = self.param['previous_frame_num']
        following_num = self.param['following_frame_num']
        self.detection_results = [[]] * (previous_num + following_num)
        self.preserve_results = [[]] * (previous_num + following_num)
        self.tube = []
        self.count = 0

    def main_process(self):
        """ Main function of dyda component. """

        current_index = self.param['previous_frame_num']
        input_data = copy.deepcopy(self.input_data)

        unpackage = False
        if isinstance(input_data, list):
            if len(input_data) == 1:
                input_data = copy.deepcopy(input_data[0])
                unpackage = True
            else:
                self.terminate_flag = True
                self.logger.error(
                    "input_data supports list of "
                    "1 element only")
                return
        input_data['filename_ori'] = input_data['filename']
        input_data['filename'] = str(self.count).zfill(8)
        self.count += 1

        if 'annotations' not in input_data.keys():
            self.detection_results.append([])
            self.preserve_results.append([])
        else:
            target_metadata = lab_tools.extract_target_class(
                copy.deepcopy(input_data), self.param_target)
            nms_annotations = lab_tools.nms_with_confidence(
                target_metadata['annotations'],
                self.param['nms_overlap_th'])

            target_metadata['annotations'] = nms_annotations

            self.detection_results.append(target_metadata)

            preserve_metadata = lab_tools.delete_target_value(
                copy.deepcopy(input_data),
                'label', self.param_target)

            self.preserve_results.append(preserve_metadata)

        if not self.detection_results[current_index] == []:
            box_number = len(
                self.detection_results[current_index]['annotations'])
            filename = self.detection_results[current_index]['filename']

            match_result_all = self.match_by_overlap_ratio_all(
                self.detection_results,
                current_index)

            tubelet_all = self.find_tubelet(
                match_result_all,
                box_number)

            self.update_tube(
                self.tube,
                tubelet_all,
                filename,
                self.param['tubelet_score_th'])

        self.remove_no_track_id()
        results = self.detection_results[self.param['previous_frame_num']]

        if results == []:
            results = input_data
        else:
            preserve_data = self.preserve_results[
                self.param['previous_frame_num']]['annotations']
            results['annotations'] += preserve_data
        for anno in results['annotations']:
            if 'track_id' not in anno.keys():
                anno['track_id'] = -1

        self.detection_results.pop(0)
        self.preserve_results.pop(0)
        self.results = self.match_output_format(results)
        self.results['filename'] = self.results.pop('filename_ori')

        if unpackage:
            self.results = [self.results]

    def match_output_format(self, results):
        """ Modify output results to meet to output format for
            parking lot solution.
        """

        del_list = ['track_score', 'lab_info']
        add_list = [['id', 0]]
        if not results == []:
            if 'annotations' in results.keys():
                annotations = results['annotations']
                for i in range(len(annotations)):
                    data = annotations[i]
                    for ai in add_list:
                        if not ai[0] in data.keys():
                            data[ai[0]] = ai[1]
                    for di in del_list:
                        if di in data.keys():
                            del data[di]
        return results

    def post_process(self):
        """ Post_process of dyda component. """

        output_parent_folder = self.lab_output_folder
        tools.check_dir(output_parent_folder)
        output_folder = os.path.join(
            output_parent_folder,
            self.__class__.__name__)
        tools.check_dir(output_folder)

        if not self.results == []:
            out_filename = os.path.join(
                output_folder,
                tools.replace_extension(
                    self.results['filename'],
                    'json'))
            tools.write_json(self.results, out_filename)

    def update_tube(
            self,
            tube,
            tubelet_all,
            filename,
            tubelet_score_th):
        """ Append tubelet to tube if score of tubelet is higher than
            tubelet_score_th

        @param tube: boxes checked from the beginning frame to current frame
        @param tubelet_all: list of tubelet
        @param filename: filename of current detection result
        @param tubelet_score_th: threshold of tubelet score

        """

        for box_index_now in range(len(tubelet_all)):
            tubelet = tubelet_all[box_index_now]
            score = tubelet['weighted_score']
            if score > tubelet_score_th:
                tube_index_now = -1
                box_index_match = -1
                for tube_index in range(len(tube)):
                    check = self.check_tubelet_belongs_to_tube(
                        tube[tube_index],
                        tubelet)
                    if check:
                        tube_index_now = tube_index
                        break
                box = {
                    'filename': filename,
                    'box_index': box_index_now,
                    'overlap_ratio': 0}
                tubelet['boxes'].append(box)
                # old tube
                if not tube_index_now == -1:
                    for match_index in range(len(tubelet['boxes'])):
                        box = tubelet['boxes'][match_index]
                        filename_matched = box['filename']
                        box_index_matched = box['box_index']
                        overlap_ratio_now = box['overlap_ratio']
                        [check, tube[tube_index_now]] = \
                            self.update_box_in_tube(tube[tube_index_now], box)
                        if check is False and box_index_matched > -1:
                            self.add_track_id(
                                filename_matched,
                                tube_index_now + 1,
                                box_index_matched,
                                overlap_ratio_now)
                            tube[tube_index_now].append({
                                'filename': filename_matched,
                                'box_index': box_index_matched,
                                'overlap_ratio': overlap_ratio_now})
                # new tube
                else:
                    tube.append([])
                    for match_index in range(0, len(tubelet['boxes'])):
                        box = tubelet['boxes'][match_index]
                        if not box['box_index'] == -1:
                            self.add_track_id(
                                box['filename'],
                                len(tube),
                                box['box_index'],
                                box['overlap_ratio'])
                            tube[tube_index_now].append({
                                'filename': box['filename'],
                                'box_index': box['box_index'],
                                'overlap_ratio': box['overlap_ratio']})

    def update_box_in_tube(
            self,
            tube,
            box):
        """ Update bounding box in tube with higher overlap ratio one

        @param tube: boxes checked from the beginning frame to current frame
        @param box: {
            'filename': filename of detection result
            'box_index': index of bounding box in the detection result
            'overlap_ratio': overlap ratio between this bounding box
                             and matched bounding box in current frame
        }

        @return check: true if any bounding box in the detection results
                       belongs to the tube
        @return tube: updated tube

        """

        check = False
        for box_index in range(len(tube)):
            if tube[box_index]['filename'] == box['filename']:
                if box['overlap_ratio'] > tube[box_index]['overlap_ratio']:
                    tube[box_index]['overlap_ratio'] = box['overlap_ratio']
                    tube[box_index]['box_index'] = box['box_index']
                check = True
        return(check, tube)

    def remove_no_track_id(self):
        """ Remove bounding box without track id from detection_results[0]

        """

        if not self.detection_results[0] == []:
            result = self.detection_results[0]['annotations']
            for i in range(len(result) - 1, -1, -1):
                if 'track_id' not in result[i].keys():
                    result.pop(i)
            self.detection_results[0]['annotations'] = result

    def add_track_id(
            self,
            filename,
            track_id,
            box_index,
            track_score):
        """ Add track id to detection result

        @param filename: filename of detection result
        @param track_id: track id given by this tracking algorithm
        @param box_index: index of bounding box added track id
        @param track_score: track score given by this tracking algorithm
                            which is weighted sum of overlap ratio

        """

        results = self.detection_results
        for i in range(len(results)):
            if not isinstance(results[i], dict):
                continue
            if len(results[i]['annotations']) == 0:
                continue
            if 'filename' in results[i].keys():
                if results[i]['filename'] == filename:
                    if len(results[i]['annotations']) <= box_index:
                        continue
                    results[i]['annotations'][box_index]['track_id'] = track_id
                    results[i]['annotations'][box_index]['track_score']\
                        = track_score
        self.detection_results = results

    def check_tubelet_belongs_to_tube(
            self,
            tube,
            tubelet):
        """ Check if tubelet belongs to tube

        @param tube: boxes checked from the beginning frame to current frame
        @param tubelet: matched boxes across several frames

        @return check: true if tubelet belongs to tube when tube and tubelet
                       have same bounding box in the same frame

        """

        check = False
        for point_index in range(len(tube)):
            for box_index in range(len(tubelet['boxes']) - 1, -1, -1):
                filename_now = tubelet['boxes'][box_index]['filename']
                box_index_now = tubelet['boxes'][box_index]['box_index']
                if tube[point_index]['filename'] == filename_now and \
                   tube[point_index]['box_index'] == box_index_now:
                    check = True
        return(check)

    def find_tubelet(
            self,
            match_result_all,
            box_number,
            temporal_decay=1.0):
        """ Find tubelet for every bounding boxe in current detection result

        @param match_result_all: list of matching results
        @param box_number: number of bounding boxes n
                           in current detection result
        @param temporal_decay: weighting of overlap ratio =
                               temporal_decay**(absolute difference of index)

        @return tubelet_all: list of tubelet with length n
                             wrt every bounding boxe
                             in current detection result
                tubelet: {
                    'boxes': list of bounding boxes across difference frames
                    'weighted_score': weighted sum of overlap ratio
                }

        """

        tubelet_all = []
        for this_box in range(box_number):
            tubelet = {
                'boxes': [],
                'weighted_score': 0}
            match_number = 0
            for mi in range(len(match_result_all)):
                temporal_diff = match_result_all[mi]['temporal_diff']
                match_result = match_result_all[mi]['match_result']
                if not match_result == []:
                    match_index_1 = match_result['match_index_1']
                    match_index_2 = match_result['match_index_2']
                    overlap_ratio = match_result['overlap_ratio']
                    if this_box in match_index_1:
                        idx = match_index_1.index(this_box)
                        box = {
                            'filename': match_result['filename_2'],
                            'box_index': match_index_2[idx],
                            'overlap_ratio': overlap_ratio[idx]}
                        tubelet['boxes'].append(box)
                        match_number = match_number + 1
                    else:
                        box = {
                            'filename': match_result['filename_2'],
                            'box_index': -1,
                            'overlap_ratio': 0}
                        tubelet['boxes'].append(box)
                    weight = temporal_decay**(temporal_diff)
                    overlap_ratio_now = box['overlap_ratio']
                    tubelet['weighted_score'] += overlap_ratio_now * weight
            tubelet_all.append(tubelet)
        return(tubelet_all)

    def match_by_overlap_ratio_all(
            self,
            detection_results,
            basis_index):
        """ Match all detection results to current detection result
            with basis_index

        @param detection_results: list of detection results
        @param basis_index: index of current detection result as matching basis

        @return match_result_all: list of matching results
        {
            'match_result': {
                'match_index_1': list of bounding box index
                                 in detection_results[basis_index]
                'match_index_2': list of bounding box index
                                 in detection_results[i]
                'overlap_ratio': list of overlap_ratio
            'temporal_diff': absolute difference between basis_index and
                             other index i
            }
        }

        """

        match_result_all = []
        for i in range(len(detection_results)):
            if i == basis_index:
                continue
            if detection_results[i] == []:
                match_result_all.append({
                    'temporal_diff': abs(i - basis_index),
                    'match_result': []})
            else:
                match_result_all.append({
                    'temporal_diff': abs(i - basis_index),
                    'match_result': self.match_by_overlap_ratio(
                        detection_results[basis_index]['annotations'],
                        detection_results[i]['annotations'],
                        self.param_overlap_ratio_th)})
                match_result_all[-1]['match_result']['filename_1'] = \
                    detection_results[basis_index]['filename']
                match_result_all[-1]['match_result']['filename_2'] = \
                    detection_results[i]['filename']
        return(match_result_all)

    def match_by_overlap_ratio(
            self,
            detection_result_1,
            detection_result_2,
            overlap_ratio_th):
        """ One to one bounding boxes matching according to overlap ratio

        @param detection_result_1: annotations in detection result
        @param detection_result_2: annotations in detection result
        @param overlap_ratio_th: only overlap ratio > overlap_ratio_th matched

        @return match_result: {
            'match_index_1': list of bounding box index in detection_result_1
            'match_index_2': list of bounding box index in detection_result_2
            'overlap_ratio': list of overlap_ratio
        }

        """

        match_result = {
            'match_index_1': [],
            'match_index_2': [],
            'overlap_ratio': []
        }

        overlap_ratio_all = []
        number_1 = len(detection_result_1)

        number_2 = len(detection_result_2)
        if number_1 == 0 or number_2 == 0:
            return(match_result)

        overlap_ratio_all = lab_tools.calculate_overlap_ratio_all(
            detection_result_1, detection_result_2)

        # score 0 when labels are different or overlap_ratio < th
        for idx_1, obj_1 in enumerate(detection_result_1):
            for idx_2, obj_2 in enumerate(detection_result_2):
                if obj_1['label'] == obj_2['label'] and \
                        overlap_ratio_all[idx_1, idx_2] > \
                        overlap_ratio_th[obj_1['label']]:
                    continue
                overlap_ratio_all[idx_1, idx_2] = 0

        # one to one match
        max_value = overlap_ratio_all.max()
        while max_value > 0:
            max_index = overlap_ratio_all.argmax()
            index_1 = int(max_index / number_2)
            index_2 = int(max_index % number_2)
            match_result['match_index_1'].append(index_1)
            match_result['match_index_2'].append(index_2)
            match_result['overlap_ratio'].append(max_value)
            overlap_ratio_all[index_1, :] = 0
            overlap_ratio_all[:, index_2] = 0
            max_value = overlap_ratio_all.max()

        return(match_result)
