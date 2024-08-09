# TasselLFANetV2
<p align="center">
  <img src="https://github.com/Ye-Sk/TasselLFANetV2/blob/master/data/infer.jpg"/>
</p>  

**The resources in this repository are implemented in this paper：**  
[___TasselLFANetV2: Exploring Vision Models Adaptation in Cross-Domain___](https://ieeexplore.ieee.org/abstract/document/10483006)

## Quantitative results
|Dataset|AP<sub>50</sub>|AP<sub>50-95</sub>|
| :----: | :----: | :----: |
|RSAD|0.981|0.603|

|Dataset|AP<sub>50</sub>|AP<sub>50-95</sub>|MAE|RMSE|R<sup>2</sup>|
| :----: | :----: | :----: | :----: | :----: | :----: |
|MTDC|0.872|0.469|3.67|5.53|0.9684|

## Installation
1. The code we implement is based on PyTorch 1.12 and Python 3.8, please refer to the file `requirements.txt` to configure the required environment.      
2. To convenient install the required environment dependencies, you can also use the following command look like this：    
~~~
$ pip install -r requirements.txt 
~~~

## Training and Data Preparation
* I have already reorganized the datasets, you just need to move them to the specified path.
#### You can download the Remote Sensing Airplane Detection (RSAD) and Maize Tassels Detection and Counting (MTDC) datasets from：
|Dataset|Baidu|Google|Source|
| :----: | :----: | :----: | :----: |
|RSAD|[Baidu](https://pan.baidu.com/s/1MfCdY824mzyZzeL-QcwJKw?pwd=89hn)|[Google](https://drive.google.com/file/d/1YCquqkTJTfyi5czpO0DG4Pwmdy4WyEiy/view?usp=sharing)|[Source](https://github.com/Ye-Sk/TasselLFANetV2)|
|MTDC|[Baidu](https://pan.baidu.com/s/16ADem84bvIkqLas-wg4kvQ?pwd=zrf6)|[Google](https://drive.google.com/file/d/1Pf7_sNJztEcMNFU5pHW5q3sEafB0po1p/view?usp=sharing)|[Source](https://github.com/poppinace/mtdc)|
* Move the dataset directly into the `data` folder, the correct data format looks like this：
~~~
$./data/RSAD (or MTDC)
├──── train
│    ├──── images
│    └──── labels
├──── test
│    ├──── images
│    └──── labels
~~~
* Run the following command to start training on the RSAD/MTDC dataset：
~~~
$ python train.py --data config/RSAD.yaml    # train RSAD
                         config/MTDC.yaml    # train MTDC
~~~
## Evaluation and Inference
* Move your trained `last.pt` model to the `data/weights` directory, the correct data format looks like this：
~~~
$./data/weights
├──── last.pt
~~~

* Run the following command to evaluate the results on the RSAD/MTDC dataset： 
~~~
$ python val.py --data config/RSAD.yaml    # eval RSAD
                       config/MTDC.yaml    # eval MTDC
~~~

* Run the following command to evaluate the counting performance on MTDC：
~~~
$ python val.py --task count --data config/MTDC.yaml    # count maize tassels
~~~

* Run the following command on a variety of sources：
~~~
$ python infer.py --save-img --source (your source path (file/dir/URL/0(webcam))) --data config/RSAD.yaml    # detect aiplanes
                                                                                         config/MTDC.yaml    # detect maize tassels
~~~

## Build your own dataset
**To train your own datasets on this framework, we recommend that :**  
1. Annotate your data with the image annotation tool [LabelIMG](https://github.com/heartexlabs/labelImg) to generate `.txt` labels in YOLO format.   
2. Refer to the `config/RSAD.yaml` example to configure your own hyperparameters file. 
3. Based on the `train.py` code example configure your own training parameters.

## Citation
#### If you find this work or code useful for your research, please cite this, Thank you!🤗
~~~
@ARTICLE{10483006,
  title={TasselLFANetV2: Exploring Vision Models Adaptation in Cross-Domain}, 
  author={Yu, Zhenghong and Ye, Jianxiong and Liufu, Shengjie and Lu, Dunlu and Zhou, Huabing},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  year={2024},
  volume={21},
  pages={1-5},
  doi={10.1109/LGRS.2024.3382871}
}
~~~
