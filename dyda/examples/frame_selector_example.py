from dyda.components import frame_reader
from dyda.components import frame_selector
from dt42lab.core import image
import os


def main():
    zero_fill = 6
    in_folder = '/home/shared/DT42/fall/Fall_CCTV_01/frame/'
    selector_result_folder = \
        '/home/shared/DT42/fall/Fall_CCTV_01/frame_selector_results/'

    # initialization
    # frame_reader
    frame_reader_ = frame_reader.FrameReader()
    # frame_selector
    # uncommand one of the following method
    #frame_selector_ = frame_selector.FrameSelectorSsimFirst()
    #frame_selector_ = frame_selector.FrameSelectorDownsampleFirst()
    frame_selector_ = frame_selector.FrameSelectorDownsampleMedian()

    for fi in range(1, len(os.listdir(in_folder)) + 1):
        fns = str(fi).zfill(zero_fill)
        image_filename = fns + '.png'
        image_file = os.path.join(in_folder, image_filename)
        # execution
        # frame_reader
        input_data = frame_reader_.run(image_file)
        # frame selector
        metadata = {
            'filename': image_filename,
            'folder': in_folder}
        selector_results = frame_selector_.run(
            input_data,
            metadata,
            selector_result_folder)
        if selector_results['is_key']:
            # classifier
            # determinater
            pass

if __name__ == "__main__":
    main()
