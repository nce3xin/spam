1. 超参数在hyperparams.py文件中。

2. 若参数CNN_mapping设置为False, 直接运行demo.py。

3. 若参数CNN_mapping设置为True，
    首先，将hyperparams.py里的MODEL设为CNN，先训练出对应的CNN模型，保存下来（已设置保存在models_saved文件夹下）。
    然后，更改loaddata/load_dataset.py第41行CNN_model_pt的值，用来加载CNN模型。

    注意，实验时，GPU和CPU不要混用，如果是用GPU训练出来的CNN结果，那么训练LSTM等其他模型时，也要用GPU。因为GPU的结果不能在CPU上跑。

4. 设置归一化时，
    如果要设置min-max归一化，则把hyperparams.py里的normalization 和 min_max_scaler同时置True，standard_scale设为False。
    如果要设置标准化，则把hyperparams.py里的normalization 和 standard_scale同时置True，min_max_scaler设为True。

