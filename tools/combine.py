if __name__ == "__main__":
    datadir = "/root/commonfile/TCTAnnotatedData/HK_TCTAnnotated20200303/"
    data, classes_name_to_id, id_reflect = coco_TCTInit()
    data, img_name_to_id = coco_getImgs(data, datadir)

    for folder in os.listdir(datadir):
        curdir = datadir + folder
        labelfile = "{}/{}".format(curdir, os.listdir(curdir)[1])
