1. ��������hyperparams.py�ļ��С�

2. ������CNN_mapping����ΪFalse, ֱ������demo.py��

3. ������CNN_mapping����ΪTrue��
    ���ȣ���hyperparams.py���MODEL��ΪCNN����ѵ������Ӧ��CNNģ�ͣ����������������ñ�����models_saved�ļ����£���
    Ȼ�󣬸���loaddata/load_dataset.py��41��CNN_model_pt��ֵ����������CNNģ�͡�

    ע�⣬ʵ��ʱ��GPU��CPU��Ҫ���ã��������GPUѵ��������CNN�������ôѵ��LSTM������ģ��ʱ��ҲҪ��GPU����ΪGPU�Ľ��������CPU���ܡ�

4. ���ù�һ��ʱ��
    ���Ҫ����min-max��һ�������hyperparams.py���normalization �� min_max_scalerͬʱ��True��standard_scale��ΪFalse��
    ���Ҫ���ñ�׼�������hyperparams.py���normalization �� standard_scaleͬʱ��True��min_max_scaler��ΪTrue��

