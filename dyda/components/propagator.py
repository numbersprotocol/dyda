import os
import copy
from dt42lab.core import data
from dt42lab.core import tools
from dt42lab.core import boxes
from dyda.core import propagator_base


class PropagatorInterpolate(propagator_base.PropagatorBase):
    """The inferencer result is propagated by interpolation
       with two nearest keyframes.

    """

    def __init__(self, dyda_config_path=''):
        super(PropagatorInterpolate, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__

        self.name_list = []
        self.data_list = []
        self.result_list = []

    def main_process(self):

        base_name = self.metadata[0]
        inferencer_result = self.input_data[0]
        self.name_list.append(base_name)
        if not inferencer_result == []:
            self.data_list.append(inferencer_result)
        if len(self.data_list) == 2:  # two nearest keyframes
            start_data = self.data_list[0]
            start_name = start_data['filename'].split('.')[0]
            start_index = self.name_list.index(start_name)
            end_data = self.data_list[1]
            end_name = end_data['filename'].split('.')[0]
            end_index = self.name_list.index(end_name)
            if len(start_data['filename'].split('.')) > 1:
                file_type = start_data['filename'].split('.')[1]
            else:
                file_type = ''
            for inter_index in range(start_index, end_index):
                inter_result = self.interpolate_by_track_id(
                    start_index,
                    start_data,
                    end_index,
                    end_data,
                    inter_index)
                inter_name = tools.replace_extension(
                    self.name_list[inter_index],
                    file_type)
                inter_result['folder'] = start_data['folder']
                inter_result['size'] = start_data['size']
                inter_result['filename'] = inter_name
                self.result_list.append(inter_result)
            self.name_list = self.name_list[end_index:]
            self.data_list.pop(0)
        if len(self.result_list) > 0:
            self.results = self.result_list[0]
            self.result_list.pop(0)
        else:
            self.results = []

    def interpolate_by_track_id(
            self,
            start_index,
            start_data,
            end_index,
            end_data,
            inter_index):
        inter_result = {'annotations': []}
        start_annotations = copy.deepcopy(start_data['annotations'])
        inter_annotations = []
        for i in range(len(start_annotations)):
            if 'track_id' in start_annotations[i].keys():
                target_id = start_annotations[i]['track_id']
                start_bb = boxes.extract_target_value(
                    copy.deepcopy(start_data),
                    'track_id',
                    target_id)['annotations']
                end_bb = boxes.extract_target_value(
                    copy.deepcopy(end_data),
                    'track_id',
                    target_id)['annotations']
                if target_id > 0 and \
                   len(start_bb) > 0 and \
                   len(end_bb) > 0:
                    inter_data = boxes.box_interpolate(
                        start_index,
                        start_bb[0],
                        end_index,
                        end_bb[0],
                        inter_index
                    )
                    inter_data['track_id'] = target_id
                    inter_annotations.append(inter_data)
        inter_result['annotations'] = inter_annotations
        return(inter_result)

    def post_process(self):
        output_parent_folder = self.lab_output_folder
        tools.check_dir(output_parent_folder)
        output_folder = os.path.join(
            output_parent_folder,
            self.__class__.__name__)
        tools.check_dir(output_folder)

        if len(self.result_list) > 0:
            for i in range(len(self.result_list)):
                result = self.result_list[i]
                out_filename = os.path.join(
                    output_folder,
                    tools.replace_extension(result['filename'], 'json'))
                data.write_json(result, out_filename)


class PropagatorDirect(propagator_base.PropagatorBase):
    """The inferencer result is propagated directly from reference
       data to base data with only filename changed.

    """

    def __init__(self, dyda_config_path=''):
        super(PropagatorDirect, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__

        self.result_list = []

    def main_process(self):

        frame_selector_result = self.input_data[0]
        inferencer_result = self.input_data[1]

        ref_data_name = frame_selector_result['ref_data_name']
        base_name = frame_selector_result['base_name']

        self.result_list.append(inferencer_result)

        found = False
        # The (-1)s below means process the list from bottom to top
        # in order to make sure the index is unchanged after pop some elements.
        for i in range(len(self.result_list) - 1, -1, -1):
            if found:
                self.result_list.pop(i)
            elif 'filename' in self.result_list[i].keys():
                filename = tools.remove_extension(
                    self.result_list[i]['filename'])
                if filename == ref_data_name:
                    found = True
                    self.results = copy.deepcopy(self.result_list[i])
                    self.results['filename'] = base_name


class PropagatorLpr(propagator_base.PropagatorBase):
    """The lpr inferencer result is propagated backward.

    """

    def __init__(self, dyda_config_path=''):
        super(PropagatorLpr, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.results = []
        self.track_list = []

    def main_process(self):

        self.results = []

        # return default results when pipeline status is 0
        if self.pipeline_status == 0:
            return

        classifier_result = copy.deepcopy(self.input_data[0])
        lpr_result = self.input_data[1]
        lpr_now = ''
        for i, res in enumerate(classifier_result):
            lpr_now += res['annotations'][0]['label']
            del res['annotations'][0]['labinfo']
            res['folder'] = self.snapshot_folder.replace(
                self.__class__.__name__,
                'DeterminatorCharacter/output_data')
            res['filename'] = self.metadata[0] + '.jpg.' + str(i)
        for anno in lpr_result['annotations']:
            if 'track_id' not in anno.keys():
                continue
            idx = max(1, anno['track_id'])
            while idx >= len(self.track_list):
                self.loop_lpr_propagate()
                self.track_list.append([])
            self.track_list[idx].append({
                'basename': self.metadata[0],
                'lpr_now': lpr_now,
                'lpr_final': anno['lpr'],
                'result': classifier_result})
        if self.metadata[0] == 'last_frame':
            self.loop_lpr_propagate()

    def loop_lpr_propagate(self):
        if len(self.track_list) == 0:
            return
        if len(self.track_list[-1]) == 0:
            return
        lpr_final = self.track_list[-1][-1]['lpr_final']
        for data in self.track_list[-1]:
            lpr_new = self.lpr_propagate(data['lpr_now'], lpr_final)
            for i in range(min(len(lpr_new), len(data['result']))):
                if lpr_new[i] in ['', '*']:
                    data['result'][i]['annotations'][0]['label'] = 'unknown'
                else:
                    data['result'][i]['annotations'][0]['label'] = lpr_new[i]
            for anno in data['result']:
                anno['annotations'][0]['type'] = 'auto-label'
            self.results.extend(data['result'])

    def lpr_propagate(self, lpr_now, lpr_final):

        len_now = len(lpr_now)
        len_pre = len(lpr_final)
        lpr_now_new = '*' * (len_pre - 1) * 2 + \
            lpr_now + '*' * (len_pre - 1)
        lpr_final_new = '*' * (len_now - 1) + \
            lpr_final + '*' * (len_now - 1)

        match_max = 0
        idx_max = 0
        for pi in range(len(lpr_now_new) - len_now + 1):
            match = []
            lpr = ''
            score = []
            for qi in range(len(lpr_final_new)):
                idx_now = pi + qi
                idx_pre = qi
                if idx_pre < 0 or idx_pre >= len(lpr_final_new):
                    continue
                if idx_now < 0 or idx_now >= len(lpr_now_new):
                    continue
                if lpr_now_new[idx_now] == '*' and \
                        lpr_final_new[idx_pre] == '*':
                    match.append(False)
                elif lpr_now_new[idx_now] == lpr_final_new[idx_pre]:
                    match.append(True)
                else:
                    match.append(False)
            if sum(match) > match_max:
                match_max = sum(match)
                idx_max = pi
        idx = 2 * len_pre - 2 - idx_max - len_now + 1
        return(lpr_final[idx:])
