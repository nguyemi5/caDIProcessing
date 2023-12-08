import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")

training_param = {}
generator_param = {}
network_param = {}
generator_test_param = {}

train_path = "\\\\neurodata2\\Large data\\JanaDataStuff\\deepInterpolation\\json_sets\\training_data_50files_1000frames.json"
val_path = "\\\\neurodata2\\Large data\\JanaDataStuff\\deepInterpolation\\json_sets\\validation_data_5files_10frames.json"

generator_param["type"] = "generator"
generator_param["name"] = "MovieJSONGenerator"
generator_param["pre_frame"] = 15
generator_param["post_frame"] = 15
generator_param["batch_size"] = 10
generator_param["train_path"] = train_path
generator_param["steps_per_epoch"] = 5

generator_test_param["type"] = "generator"
generator_test_param["name"] = "MovieJSONGenerator"
generator_test_param["pre_frame"] = 15
generator_test_param["post_frame"] = 15
generator_test_param["batch_size"] = 10
generator_test_param["train_path"] = val_path
generator_test_param["steps_per_epoch"] = -1

network_param["type"] = "network"
network_param["name"] = "unet_single_1024"

training_param["type"] = "trainer"
training_param["name"] = "core_trainer"
training_param["run_uid"] = run_uid
training_param["batch_size"] = generator_param["batch_size"]
training_param["steps_per_epoch"] = 5
training_param["period_save"] = 50
training_param["nb_gpus"] = 1
# training_param["apply_learning_decay"] = 1
training_param["nb_times_through_data"] = 1
# training_param["initial_learning_rate"] = 0.0005
training_param["learning_rate"] = 0.0005
# training_param["epochs_drop"] = 50
training_param["loss"] = "mean_absolute_error"
training_param["model_string"] = network_param["name"] + "_" + training_param["loss"]
training_param['caching_validation'] = False

jobdir = (
    "C:/Users/MinhThao/Desktop/NeuroCode/jn/deepInterpolation/out"
    + training_param["model_string"]
    + "_"
    + "pre_"
    + str(generator_param["pre_frame"])
    + "post_"
    + str(generator_param["post_frame"])
    + run_uid
)

training_param["output_dir"] = jobdir

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

path_training = os.path.join(jobdir, "training.json")
json_obj = JsonSaver(training_param)
json_obj.save_json(path_training)

path_generator = os.path.join(jobdir, "generator.json")
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

path_test_generator = os.path.join(jobdir, "test_generator.json")
json_obj = JsonSaver(generator_test_param)
json_obj.save_json(path_test_generator)

path_network = os.path.join(jobdir, "network.json")
json_obj = JsonSaver(network_param)
json_obj.save_json(path_network)

generator_obj = ClassLoader(path_generator)
generator_test_obj = ClassLoader(path_test_generator)

network_obj = ClassLoader(path_network)
trainer_obj = ClassLoader(path_training)

train_generator = generator_obj.find_and_build()(path_generator)
test_generator = generator_test_obj.find_and_build()(path_test_generator)

network_callback = network_obj.find_and_build()(path_network)

training_class = trainer_obj.find_and_build()(
    train_generator, test_generator, network_callback, path_training
)

training_class.run()

training_class.finalize()
