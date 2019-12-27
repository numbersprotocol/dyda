{
    "BinaryDataReader":
    {
    },
    "FrameReader":
    {
    },
    "DetectorYOLO":
    {
        "lib_path": "/usr/lib/libdarknet.so",
        "net_cfg": "/home/shared/DT42/LPR/plate_model/yolo-voc.cfg",
        "net_weights": "/home/shared/DT42/LPR/plate_model/yolo-voc_final.weights",
        "net_meta": "/home/shared/DT42/LPR/plate_model/dt42.data",
        "thresh": 0.3,
        "hier_thresh": 0.5,
        "nms": 0.1
    },
    "FrameSelectorDownsampleFirst": {
        "interval": 1
    },
    "DeterminatorTargetLabel": {
        "target": "licence_plate"
    },
    "DeterminatorByRoi": {
        "top": 0,
        "bottom": 1079,
        "left": 0,
        "right": 1790,
        "threshold": 0
    },
    "DeterminatorCharacter": {
        "char_number": [7],
        "plate_ext_top": -0.1,
        "plate_ext_bottom": -0.1,
        "plate_ext_left": 0.1,
        "plate_ext_right": 0.1,
        "char_size": 128,
        "output_size": 50,
        "local_bin_thre": 15,
        "projection_v0_length_ratio_thre": 0.5,
        "projection_v0_percentile_thre": 0.95,
        "projection_h0_length_ratio_thre": 0.5,
        "projection_h0_percentile_thre": 0.9,
        "projection_h1_length_ratio_thre": 0.3,
        "projection_h1_percentile_thre": 85,
        "projection_v1_length_ratio_thre": 0.03,
        "projection_v1_percentile_thre": 85,
        "angle_min": -5,
        "angle_max": 6,
        "angle_gap": 5,
        "shift_ratio_min_y": -1,
        "shift_ratio_max_y": 5,
        "shift_ratio_min_x": 0,
        "shift_ratio_max_x": 1,
        "char_ratio_min": 1.6,
        "char_ratio_max": 2.6,
        "bin_thre_min": -20,
        "bin_thre_max": 35,
        "bin_thre_gap": 5
    },
    "ClassifierMobileNet": {
        "model_file": "/home/shared/model_zoo/char_mobilenet_1.0_128_v2/output_graph.pb",
        "label_file": "/home/shared/model_zoo/char_mobilenet_1.0_128_v2/output_labels.txt",
        "input_height": 128,
        "input_width": 128,
        "input_mean": 128,
        "input_std": 128,
        "input_layer": "input",
        "convert_to_rgb": true,
        "output_layer": "final_result",
        "ftype": "jpg",
        "gpu_options": {
            "allow_growth": true,
            "visible_device_list": "0",
            "per_process_gpu_memory_fraction": 0.3
        }
    },
    "OutputGeneratorLpr": {
        "confidence_thre": 0.99999,
        "match_bit_thre": 4,
        "continuous": true
    },
    "PadImageProcessor":
    {
        "out_w": 416,
        "out_h": 416
    },
    "TrackerByOverlapRatio": {
        "target": ["licence_plate"],
        "previous_frame_num": 7,
        "following_frame_num": 0,
        "nms_overlap_th": 0.3,
        "tubelet_score_th": 0.7
    },
    "PatchImageProcessor":
    {
        "patch_color": [0, 0, 255],
        "patch_line_width": 6,
        "text_space": 30,
        "key_to_patch": ["track_id", "lpr"],
        "snapshot_with_counter": false
    }
}
