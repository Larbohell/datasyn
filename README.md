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

### Datasets
GTSRB: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads

### Resources 
Traffic sign recognition with tensorflow-tutorial:
https://github.com/waleedka/traffic-signs-tensorflow/blob/master/notebook1.ipynb

More detailed description of the tutorial above:
https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6#.cu8mczadx

Machine learning/CNN guide:
https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.r6cklogiy

Bedre classifier med pre-processing
http://jokla.me/robotics/traffic-signs/
