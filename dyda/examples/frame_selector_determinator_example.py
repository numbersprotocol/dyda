import traceback
import os
import sys

from dyda.components import frame_reader
from dyda.components import frame_selector
from dyda.components import determinator
from dt42lab.core import image


def main():
    zero_fill = 8
    in_folder = '/home/shared/DT42/dyda_test/frame/'
    selector_result_folder = \
        '/home/shared/DT42/dyda_test/selector_results/'
    classifier_result_folder = \
        '/home/shared/DT42/dyda_test/classifier_results/'

    # initialization
    # frame_reader
    frame_reader_ = frame_reader.FrameReader()
    # frame_selector
    # uncommand one of the following method
    #frame_selector_ = frame_selector.FrameSelectorSsimFirst()
    #frame_selector_ = frame_selector.FrameSelectorDownsampleFirst()
    frame_selector_ = frame_selector.FrameSelectorDownsampleMedian()
    # determinator
    determinator_ = determinator.DeterminatorConfidenceThreshold()

    fpaths = image.get_images(in_folder)
    for fi in range(1, len(fpaths) + 1):
        try:
            fns = str(fi).zfill(zero_fill)
            image_filename = fpaths[fi - 1]
            #print(image_filename, fi, fns)
            # execution
            # frame_reader
            data = frame_reader_.run(image_filename)
            # frame selector
            metadata = {
                'filename': image_filename,
                'folder': in_folder}
            selector_results = frame_selector_.run(
                data,
                metadata,
                selector_result_folder)
            if selector_results['is_key']:
                # classifier
                # determinater
                classifier_results_filename = os.path.join(
                    classifier_result_folder,
                    fns + '.json')
                determinator_.run(classifier_results_filename)
        except Exception:
            traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    main()
