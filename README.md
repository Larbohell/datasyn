# Datasyn-prosjekt

## How-to kjøre object detection med tensorbox
https://github.com/cheind/TensorBox/tree/develop

1. `cd tensorbox`
2. `python train.py --hypes hypes/overfeat_rezoom.json --gpu 0 --logdir output`
* Evt. `python train.py --hypes hypes/overfeat_rezoom.json --gpu 0 --logdir output --max_iter <max number of iterations>`
3) Evaluation: `python evaluate.py --weights output/overfeat_rezoom_<correct_folder>/save.ckpt-130000 --test_boxes data/brainwash/val_boxes.json`

## To do
* Skrive til fil, så nettverket ikke må trenes hver gang


### Forbedringsmuligheter

* Bildesegmentering for å finne skilt
* Klassisk bildesegmentering som input
    * Histogram-equalization
* Fartsgrenseskilt: Skille mellom forskjellige fartsgrenser, kanskje konvertere bildet -> tall


### Resources 
Traffic sign recognition with tensorflow-tutorial:
https://github.com/waleedka/traffic-signs-tensorflow/blob/master/notebook1.ipynb

More detailed description of the tutorial above:
https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6#.cu8mczadx

Machine learning/CNN guide:
https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.r6cklogiy

Walkthrough of tensor flow with small ANN. See task 2 (ML-course assignment 4 - Johan):
https://drive.google.com/file/d/0B1EcuNVaOt3QeU5xb2xzNUxpSWc/view?usp=sharing
