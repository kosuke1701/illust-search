from argparse import ArgumentParser
import sqlite3
import numpy as np
import io
import os

from tqdm import tqdm

from animecv.util import load_image, from_PIL_to_cv, \
    add_bounding_box, write_image
from animecv.character_identification import Res18_CharacterIdentifier_BBox
from animecv.object_detection import FaceDetector_EfficientDet

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = 1000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Following codes which define `array` type in sqlite3 is copied from the following Stack Overflow:
# https://stackoverflow.com/questions/18621513
# question by:
# Joe Flip (https://stackoverflow.com/users/1715453/joe-flip)
# answered by:
# unutbu (https://stackoverflow.com/users/190597/unutbu)
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())
def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)
# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--target-fn", type=str, help="File which contains list of illust id and image filenames in TSV format.")
    parser.add_argument("--image-root", type=str, help="Root directory where image files are stored.")
    parser.add_argument("--cuda", action="store_true", help="Use GPU device or not.")

    args = parser.parse_args()

    # Open SQL database
    ## [face]
    ##   id INTEGER
    ##   face INTEGER
    ##   xmin, xmax, ymin, ymax INTEGER
    ##   vector array
    DB_NAME = "vectors.sql"

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("CREATE TABLE IF NOT EXISTS face \
        (id INTEGER, face INTEGER, xmin INTEGER, \
        xmax INTEGER, ymin INTEGER, ymax INTEGER, \
        vector array)")
    c.execute("CREATE TABLE IF NOT EXISTS face_log (id INTEGER)")
    
    conn.commit()

    # Load list of processed images.
    c.execute("SELECT id FROM face_log")
    processed_ids = {row[0] for row in c.fetchall()}

    # Initialize models
    identifier = Res18_CharacterIdentifier_BBox()
    if args.cuda:
        identifier.to("cuda")
    detector = FaceDetector_EfficientDet(coef=0, use_cuda=args.cuda)

    # Process images.
    target_files = []
    with open(args.target_fn) as h:
        for line in h:
            line = line.replace("\n", "")
            row = line.split("\t")
            if len(row) == 2:
                i_illust = int(row[0])
                if i_illust in processed_ids:
                    continue
                target_files.append((i_illust, row[1])) # i_illust, filename
    

    for i, (i_illust, fn) in enumerate(tqdm(target_files)):
        fn = os.path.sep.join([args.image_root, fn])

        target_img = load_image(fn)
        target_bbox = detector.detect([target_img])
        if len(target_bbox[0]) > 0:
            target_emb, target_i_img, target_i_bbox = \
                identifier.encode_image([target_img], target_bbox)
            
            for i_emb, i_bbox in enumerate(target_i_bbox):
                x_min, y_min, x_max, y_max = \
                    map(int,target_bbox[0][i_bbox]["coordinates"])

                emb = target_emb[i_emb].detach().cpu().numpy()
                c.execute("INSERT INTO face (id, face, xmin, xmax, ymin, ymax, vector) \
                    VALUES (?, ?, ?, ?, ?, ?, ?)", \
                    (i_illust, i_bbox, x_min, x_max, y_min, y_max, emb))
        
        c.execute("INSERT INTO face_log (id) VALUES (?)", (i_illust,))

        if i%100==0:
            conn.commit()
    conn.commit()
    conn.close()

    print("Finished")
        
        