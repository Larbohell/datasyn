import json
from PIL import Image
import os

def main(json_file_path):

    json_file = open (json_file_path)
    json_string = json_file.read()
    json_data = json.loads(json_string)

    image = json_data[0]
    #for image in json_data:
    image_file_path = image['image_path']
    image_to_crop = Image.open(image_file_path)

    cropped_images = []

    image_rects = image['rects']

    for r in image_rects:
        score = r['score']

        if score > 0.0:

            wider = 0
            new_image = image_to_crop.crop((r['x1']-wider, r['y1']-wider, r['x2']+wider, r['y2']+wider))

            new_image = transform_image_to_square(new_image) #transform the image to square
            cropped_images.append(new_image)

    return cropped_images


def transform_image_to_square(img):
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