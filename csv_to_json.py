import csv
import json

FILENAME_CSV = "datasets/detection/TrainIJCNN2013/gt.txt"
FILENAME_JSON = "datasets/detection/TrainIJCNN2013/gt.json"

def main():
    csvfile = open(FILENAME_CSV, 'r')
    jsonfile = open(FILENAME_JSON, 'w')

    fieldnames = ("Filename", "Leftcol", "TopRow", "RightCol", "BottomRow", "ClassID")
    reader = csv.DictReader(csvfile, fieldnames, delimiter=";")

    lastFileName = ""
    jsontmp = dict()
    i = 0
    jsonfile.write("[")
    firstBatchWritten = False

    for row in reader:
        if row["Filename"] != lastFileName:
            # New image file
            if lastFileName != "":
                # Write current image file data to json file
                if firstBatchWritten:
                    jsonfile.write(",")
                else:
                    firstBatchWritten = True

                json.dump(jsontmp, jsonfile)

            # Start section for new file
            jsontmp.clear()
            jsontmp["image_path"] = row["Filename"]
            jsontmp["rects"] = []
            i = 0

        jsontmp["rects"].append({})

        jsontmp["rects"][i]["x1"] = row["Leftcol"]
        jsontmp["rects"][i]["x2"] = row["RightCol"]
        jsontmp["rects"][i]["y1"] = row["TopRow"]
        jsontmp["rects"][i]["y2"] = row["BottomRow"]
        i += 1
        lastFileName = row["Filename"]

    jsonfile.write(",")
    json.dump(jsontmp, jsonfile)
    jsonfile.write("]")

main()