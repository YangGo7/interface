from flask import Flask, request, Response, send_file
import uuid
import os
import sys
from temp_utils import *
from ultralytics import YOLO
import cv2
import numpy as np
import json
import shutil
from PIL import Image

app = Flask(__name__)

def get_val(keyval):
    if keyval[0] == '1':
        keyval = '2' + keyval[1]
    elif keyval[0] == '2':
        keyval = '1' + keyval[1]
    elif keyval[0] == '3':
        keyval = '4' + keyval[1]
    elif keyval[0] == '4':
        keyval = '3' + keyval[1]
    return keyval
    
@app.route('/upload', methods=['POST'])
def upload_file():
    
    #if set(request.files.keys()) != set(['pano', 'json']):
    #    return 'false'
    #print(request.form.keys())
    #print(request.get_data())
    #print(request.form['json'])
    #print(request.form['pano'])
    #print(request.files.keyes())

    pano = request.files['pano']  # 클라이언트에서 전달된 파일
    js = request.files['json']  # 클라이언트에서 전달된 파일

    #if pano is None or js is None:
    #    return 'false'

    if os.path.exists('files'):
        shutil.rmtree('files')
    os.mkdir('files')


    js.save(os.path.join('files', js.filename))  # 파일 저장
    pano.save(os.path.join('files', pano.filename))
    with open(os.path.join('files', js.filename), 'r') as f:
        info = json.load(f)

    image_path = os.path.join('files', pano.filename)
    original = cv2.imread(image_path)
    SEG = YOLO('tooth_seg.pt')
    seg_res = SEG([image_path])
    
    #print(seg_res[0].names)
    #print(dir(seg_res))

    
    for idx, coords in enumerate(seg_res[0].masks.xy):
        # keyval is teeth number
        keyval = seg_res[0].names[seg_res[0].boxes.cls[idx].item()]
        keyval = get_val(keyval)
        if keyval == '유치' or keyval =='과잉치':continue
        info["Tooth"][keyval]["status"] = 'true'
        
    #print(info)
    # CARIES
    caries_label = get_diagnosis_info(copy.deepcopy(original), seg_res, 'c')
    for v in caries_label:
        info["Tooth"][v]["caries"]='true'
    # PERIAPICAL_LESION
    lesion_label = get_diagnosis_info(copy.deepcopy(original), seg_res, 'p')
    #for v in caries_label:
    for v in lesion_label:
        info["Tooth"][v]["lesion"]='true'
    # BONE LEVEL
    pbl_label = get_bonelevel_info(copy.deepcopy(original), seg_res)
    for v in pbl_label.keys():
        info["Tooth"][v]["pdrate"] = pbl_label[v]
        if pbl_label[v] > 85:
            info["Tooth"][v]["pdlevel"] = 1
        elif pbl_label[v] > 67 and pbl_label[v] < 85:
            info["Tooth"][v]["pdlevel"] = 2
        else:
            info["Tooth"][v]["pdlevel"] = 3
    info["Pano"] = str(int(info["Pano"])+1)
    
    with open(os.path.join('files', js.filename), 'w') as f:
        print(" +++++++++++++++++++++++++++++++++ ")
        json.dump(info, f)
    
    return 'true'

@app.route('/download', methods=['GET'])
def download_file():
    filename = request.args.get('filename')
    target = os.path.join('files', filename)
    if os.path.exists(target):
        with open(os.path.join('files', filename), 'r') as f:
            info = json.load(f)

        return Response(json.dumps(info, sort_keys=False), mimetype='application/json')
    else:
        return {}

if __name__ == '__main__':
    # app.run("192.168.0.30")
    #app.run("192.168.60.185")
    app.run("0.0.0.0")
