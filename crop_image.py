import json
from PIL import Image
import os

def main():
    FILE_FORMAT = ".ppm" #Specify the format of the output images
    #FILE_FORMAT = ".jpg"

    #Specify the correct paths
    json_file_path = 'TensorBox/output/overfeat_rezoom_2017_04_18_23.35/save.ckpt-7500.val_boxes.json'
    cropped_images_location = 'TensorBox/output/overfeat_rezoom_2017_04_18_23.35/cropped_images'

    json_file = open (json_file_path)
    json_string = json_file.read()
    json_data = json.loads(json_string)

    i = 0

    for image in json_data:
        image_file_path = image['image_path']
        image_to_crop = Image.open(image_file_path)

        image_rects = image['rects']

        for r in image_rects:
            score = r['score']

            if score > 0.0:

                wider = 0
                new_image = image_to_crop.crop((r['x1']-wider, r['y1']-wider, r['x2']+wider, r['y2']+wider))
                #new_image = image_to_crop.crop((r['x1'], r['y1'], r['x2'], r['y2']))

                if not os.path.exists(cropped_images_location):
                    os.makedirs(cropped_images_location)

                new_image = pad_image_to_square(new_image) #transform the images to squares

                new_image.save(cropped_images_location + "/"+str(i)+"_score_"+str(score)+FILE_FORMAT)
                #new_image.save(cropped_images_location + "/" + str(score) + ".jpg")

                i+=1

def pad_image_to_square(img):
    longer_side = max(img.size)
    horizontal_padding = (longer_side - img.size[0]) / 2
    vertical_padding = (longer_side - img.size[1]) / 2

    square_img = img.crop(
        (
            -horizontal_padding,
            -vertical_padding,
            img.size[0] + horizontal_padding,
            img.size[1] + vertical_padding
        )

    )

    return square_img

main()