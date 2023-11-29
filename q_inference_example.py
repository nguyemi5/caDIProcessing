import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import pathlib

if __name__ == '__main__':
    generator_param = {}
    inferrence_param = {}

    # We are reusing the data generator for training here.
    generator_param["type"] = "generator"
    generator_param["name"] = "InferenceOphysGeneratorMat"
    generator_param["pre_post_frame"] = 15
    generator_param["pre_post_omission"] = 0
    generator_param[
        "steps_per_epoch"
    ] = -1
    # No steps necessary for inference as epochs are not relevant.
    # -1 deactivate it.

    # generator_param["train_path"] = os.path.join(
    #     pathlib.Path(__file__).parent.absolute(),
    #     "..",
    #     "sample_data",
    #     "M430F3File14.tif",
    # )

    # generator_param["train_path"] = "//neurodata2/Large data/JanaDataStuff/deepInterpolation/train/M430F3File14.h5"
    generator_param["train_path"] = "/storage/brno2/home/nguyenomi/file_00014_aligned_CH2outOf200001.mat"

    generator_param["batch_size"] = 1
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = -1  # -1 to go until the end.
    generator_param[
        "randomize"
    ] = 0
    # This is important to keep the order
    # and avoid the randomization used during training

    inferrence_param["type"] = "inferrence"
    inferrence_param["name"] = "core_inferrence"
    inferrence_param["use_multiprocessing"] = True

    # Replace this path to where you stored your model
    # inferrence_param[
    #     "model_path"
    # ] = "C:/Users/MinhThao/Desktop/NeuroCode/jn/deepInterpolation/models/orig_ai93.h5"

    inferrence_param[
        "model_path"
    # ] = "C:/Users/MinhThao/Desktop/NeuroCode/jn/deepInterpolation/outunet_single_1024_mean_absolute_error_pre_15post_152023_10_06_15_37/2023_10_06_15_37_unet_single_1024_mean_absolute_error_model.h5"
    # ] = "C:/Users/MinhThao/Desktop/NeuroCode/jn/deepInterpolation/outunet_single_1024_mean_absolute_error_pre_15post_152023_10_05_17_56/2023_10_05_17_56_unet_single_1024_mean_absolute_error_model.h5"
    # ] = "C:/Users/MinhThao/Desktop/NeuroCode/jn/deepInterpolation/outunet_single_1024_mean_absolute_error_pre_30post_302023_10_07_17_18/2023_10_07_17_18_unet_single_1024_mean_absolute_error-0050-0.0000.h5"
    ] = "/storage/brno2/home/nguyenomi/2023_10_06_15_37_unet_single_1024_mean_absolute_error-0050-0.0000.h5"
    # Replace this path to where you want to store your output file
    inferrence_param[
        "output_file"
    ] = "/storage/brno2/home/nguyenomi/out_M430F3File14_Pre15.h5"

    jobdir = "out"


    try:
        os.mkdir(jobdir)
    except Exception:
        print("folder already exists")

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_infer = os.path.join(jobdir, "inferrence.json")
    json_obj = JsonSaver(inferrence_param)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    inferrence_obj = ClassLoader(path_infer)
    inferrence_class = inferrence_obj.find_and_build()(path_infer,
                                                       data_generator)

    # Except this to be slow on a laptop without GPU. Inference needs
    # parallelization to be effective.
    inferrence_class.run()
