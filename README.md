# Datasyn-prosjekt

## How-to kjøre object detection med tensorbox
https://github.com/cheind/TensorBox/tree/develop

0. Download the correct datasets into TensorBox/data, as well as `inception_v1.ckpt`, from:
http://russellsstewart.com/s/tensorbox/inception_v1.ckpt

* Ville vært det samme som kjøre `download_data.sh`, som også ville gitt ansikts-datasettet 

1. `cd tensorbox`
2. `python train.py --hypes hypes/overfeat_rezoom.json --gpu 0 --logdir output`
   1. Evt. slenge på `--max_iter <max iter>` og `--save_iter <save iter>`
   2. Hvis prosessen avsluttes med ctrl c får man resultatene fra siste lagringspunkt, gitt av save_iter-variabelen. Default satt til 10
3. Evaluation: `python evaluate.py --weights output/overfeat_rezoom_<correct_folder>/save.ckpt-<numOfIterations> --test_boxes data/brainwash/val_boxes.json`

4. Finn visualisering av evauleringen i output/overfeat_rezoom_<correct_folder>/images_val_boxes<numOfIterations>

## Har gjort
* Funnet bildeklassifiseringsAI
* Funnet bildesegmenteringsAI
* Skrive/lese nettverk til/fra fil 
* Forbedre bildeklassiferingsAI med flere hidden layers i nettverket
* Integrert system for å detektere og gjenkjenne
* Preprocessed for forbedret deteksjon

## To do
* Tidstest av segmentering og klassifisering av skilt
* Laste inn hidden layers
* Finne accuracy for ulikt antall hidden layers og bestemme optimalt antall
* Flere bilder fra live video feed sendes inn og kan brukes sammen + Random Forest på resultatene fra alle bildene
* Precision/recall stats

## Verdt å notere seg
* Skiltfinneren leter ikke etter de samme skiltene som classifyeren vår

### Forbedringsmuligheter

* Bildesegmentering for å finne skilt
* Klassisk bildesegmentering som input
* Fartsgrenseskilt: Skille mellom forskjellige fartsgrenser, kanskje konvertere bildet -> tall
* Parallellisere bildesegmentering og skiltklassifisering
* Random image morphing as data augmentation for every traning epoch

** Some possible improvements:
* I would use Keras to define the network and its function ImageDataGenerator to generate augmented samples on the fly. Using more data could improve the performance of the model. In my case, I have generated an augmented dataset once, saved it on the disk and used it every time to train. It would be useful to generate randomly the dataset each time before the training.
* The confusion matrix gives us suggestions to improve the model (see section Confusion matrix). There are some classes with low precision or recall. It would be useful to try to add more data for these classes. For example, I would generate new samples for the class 19 (Dangerous curve to the left) since it has only 180 samples and the model.
* The accuracy for the training set is 0.975. This means that the model is probably underfitting a little bit. I tried to make a deeper network (adding more layers) and increasing the number of filters but it was too slow to train it using the CPU only.
* The model worked well with new images taken with my camera (100% of accuracy). It would be useful to test the model by using more complicated examples.


### Resources 
Traffic sign recognition with tensorflow-tutorial:
https://github.com/waleedka/traffic-signs-tensorflow/blob/master/notebook1.ipynb

More detailed description of the tutorial above:
https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6#.cu8mczadx

Machine learning/CNN guide:
https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.r6cklogiy

Walkthrough of tensor flow with small ANN. See task 2 (ML-course assignment 4 - Johan):
https://drive.google.com/file/d/0B1EcuNVaOt3QeU5xb2xzNUxpSWc/view?usp=sharing

Loss and accuracy in tensorflow explained:
http://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model

Tensorflow layers calculation
https://www.tensorflow.org/get_started/mnist/mechanics

Bedre classifier med pre-processing
http://jokla.me/robotics/traffic-signs/
