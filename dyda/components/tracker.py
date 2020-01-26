import os
import copy
import numpy as np
from dyda.core import tracker_base
from dyda_utils import lab_tools
from dyda_utils import tools


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

        # WORKAROUND: "% 1000" is for fixing AIKEA pipeline not working issue.
        #             Need to remove it and find a better solution.
        obj['track_id'] = self.track_id % 1000
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
