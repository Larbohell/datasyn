
import TensorBox/evaluate
import test
IMAGE = "datasets/detection/TestIJCNN2013/00103.ppm"
DETECTED_SIGNS_DIR = "output/detected_signs"
MODEL_DIR = "output/BelgiumTS/2017_04_21_19.44_300"

def main():
    #Sign detection
        #DO STUFF

    #Sign recognition
    test.main(DETECTED_SIGNS_DIR, MODEL_DIR)


main()