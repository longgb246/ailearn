# coding=utf-8

# Lint as: python3
# """Trains TFT based on a defined set of parameters.

# Uses default parameters supplied from the configs file to train a TFT model from
# scratch.

# Usage:
# python3 script_train_fixed_params {expt_name} {output_folder}

# Command line args:
#   expt_name: Name of dataset/experiment to train.
#   output_folder: Root folder in which experiment is saved


# """

import datetime as dte
import os
import sys

import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

ExperimentConfig = dl_learn.Predict.tft.expt_settings.configs.ExperimentConfig
HyperparamOptManager = dl_learn.Predict.tft.libs.hyperparam_opt.HyperparamOptManager
ModelClass = dl_learn.Predict.tft.libs.tft_model.TemporalFusionTransformer


sys.path.append('F:/Codes/ailearn/longgb/DL_Learn/Predict/tft')


def main():
    name, output_folder, use_tensorflow_with_gpu = 'traffic', 'F:/Codes/data1/tft_outputs', False

    config = ExperimentConfig(name, output_folder)
    formatter = config.make_data_formatter()

    expt_name = name
    use_gpu = use_tensorflow_with_gpu
    model_folder = os.path.join(config.model_folder, "fixed")
    data_csv_path = config.data_csv_path
    data_formatter = formatter
    use_testing_mode = True

    num_repeats = 1  # 重复次数

    # Tensorflow setup
    default_keras_session = tf.keras.backend.get_session()

    if use_gpu:
        tf_config = utils.get_default_tensorflow_config(
            tf_device="gpu", gpu_id=0)
    else:
        tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

    # Read data
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration(
    )

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Parameter overrides for testing only! Small sizes used to speed up script.
    if use_testing_mode:
        fixed_params["num_epochs"] = 1
        params["hidden_layer_size"] = 5
        train_samples, valid_samples = 100, 10

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)

    # Training -- one iteration only
    print("*** Running calibration ***")
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    train_data = model.TFTDataCache.get('train')
    train_data['inputs'].shape
    train_data['outputs'].shape
    train_data['active_entries'].shape
    train_data['time'].shape
    train_data['identifier'].shape

    model._get_active_locations(train_data['active_entries']).shape

    [[i, i+1] for i in range(3)] + [[i, i+1] for i in range(3)]

    # Training
    best_loss = np.Inf
    for _ in range(num_repeats):
        tf.reset_default_graph()
        with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
            # sess = tf.Session(config=tf_config)
            tf.keras.backend.set_session(sess)
            params = opt_manager.get_next_parameters()
            model = ModelClass(params, use_cudnn=use_gpu)
            if not model.training_data_cached():
                model.cache_batched_data(
                    train, "train", num_samples=train_samples)
                model.cache_batched_data(
                    valid, "valid", num_samples=valid_samples)
            sess.run(tf.global_variables_initializer())
            model.fit()
            val_loss = model.evaluate()
            if val_loss < best_loss:
                opt_manager.update_score(params, val_loss, model)
                best_loss = val_loss
            tf.keras.backend.set_session(default_keras_session)

    print("*** Running tests ***")
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        tf.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)
        model.load(opt_manager.hyperparam_folder)
        print("Computing best validation loss")
        val_loss = model.evaluate(valid)

        print("Computing test loss")
        output_map = model.predict(test, return_targets=True)
        targets = data_formatter.format_predictions(output_map["targets"])
        p50_forecast = data_formatter.format_predictions(output_map["p50"])
        p90_forecast = data_formatter.format_predictions(output_map["p90"])

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        p50_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(
                targets), extract_numerical_data(p50_forecast),
            0.5)
        p90_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(
                targets), extract_numerical_data(p90_forecast),
            0.9)
        tf.keras.backend.set_session(default_keras_session)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
    print("\nNormalised Quantile Loss for Test Data: P50={}, P90={}".format(
        p50_loss.mean(), p90_loss.mean()))


if __name__ == "__main__":
    main()
