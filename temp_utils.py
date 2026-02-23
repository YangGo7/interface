import copy
import glob
from ultralytics import YOLO
import numpy as np
import os
import statistics as st
import math

from PIL import Image, ImageDraw
from efficientnet_pytorch import EfficientNet
import torch
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2


seg_model = YOLO("./tooth_seg.pt")
imsize = 192
lst = glob.glob('./data/*.jpg')

model_weight = "./best_weights_b5_train3_batch32_isize192_.pth"
BONELEVEL = "./bonelevel.pt"
CEJ = "./cej.pt"
PERIAPICAL_LESION = "./best.pt"
CARIES = "./caries_det.pt"

def get_iou(b1, b2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    #b1 = [[b1[0], b1[1]], [b1[0]+b1[2], b1[1]+b1[3]]]

    bb1 = {}
    bb2 = {}
    if b1[0][0] < b1[1][0]:
        bb1['x1'] = b1[0][0]
        bb1['x2'] = b1[1][0]
    else:
        bb1['x1'] = b1[1][0]
        bb1['x2'] = b1[0][0]

    if b1[0][1] < b1[1][1]:
        bb1['y1'] = b1[0][1]
        bb1['y2'] = b1[1][1]
    else:
        bb1['y1'] = b1[1][1]
        bb1['y2'] = b1[0][1]

    if b2[0][0] < b2[1][0]:
        bb2['x1'] = b2[0][0]
        bb2['x2'] = b2[1][0]
    else:
        bb2['x1'] = b2[1][0]
        bb2['x2'] = b2[0][0]

    if b2[0][1] < b2[1][1]:
        bb2['y1'] = b2[0][1]
        bb2['y2'] = b2[1][1]
    else:
        bb2['y1'] = b2[1][1]
        bb2['y2'] = b2[0][1]


    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])


    if x_right < x_left or y_bottom < y_top:
        return 0.0, bb2

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    # iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    iou = intersection_area / bb2_area
    assert iou >= 0.0
    assert iou <= 1.0
    return iou, bb2

def get_squeeze_points(coordinates, img_path, e=1, opt=cv2.CHAIN_APPROX_TC89_L1):
    def erode(src, epoch=1):
        dst = copy.deepcopy(src)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for i in range(epoch):
            dst = cv2.erode(dst, k)

        return dst

    img = img_path
    h, w, c = img.shape
    mask = np.zeros([h, w], dtype=np.uint8)

    linearr = []

    for xy in coordinates:
        x = int(xy[0])
        y = int(xy[1])
        # mask[x][y] = 255
        linearr.append([x, y])
    
    mask = cv2.fillPoly(mask, np.int32([linearr]), (255, 255, 255))
    post_mask = cv2.fillPoly(mask, np.int32([linearr]), (255, 255, 255))

    post_mask = erode(post_mask, e)

    contour, _ = cv2.findContours(post_mask, cv2.RETR_EXTERNAL, opt)

    return contour

def get_principal_axis(cropped_img, num):

    def get_root_pos(img, th=5, opt='up'):
        most_left = 999
        most_right = 0
        if th >= len(img):
            th = len(img) - 2
        start = len(img)-1
        end = len(img) - th
        step = -1
        if opt == 'up':
            start = 0
            end = th
            step = 1

        # # in multi root case, need to control 'y' value
        # bias = start
        for col in range(start, end, step):
            for row in range(len(img[col])):
                if img[col][row] != 0:
                    if most_left >= row:
                        most_left = row
                if img[col][row] != 0:
                    if most_right <= row:
                        most_right = row

        return (int((most_left+most_right)/2), start)

    def get_crown_pos(img, num, th=15, opt='up'):

        if num[1] in ['6', '7']:
            th = 30
        if th >= len(img):
            th = len(img) - 2
        most_left = 999
        most_right = 0
        start = 0
        end = th
        if opt == 'up':
            start = len(img)-th
            end = len(img)

        for col in range(start, end):
            for row in range(len(img[col])):
                if img[col][row] != 0:
                    if most_left > row:
                        most_left = row
                if img[col][row] != 0:
                    if most_right < row:
                        most_right = row

        temp = copy.deepcopy(img)
        temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
        if opt == 'up':
            cv2.circle(temp, (int((most_left + most_right) / 2), end), 2, (0, 0, 255), 1)
        else:
            cv2.circle(temp, (int((most_left + most_right) / 2), start), 2, (0, 0, 255), 1)

        if opt == 'up':
            return (int((most_left + most_right) / 2), end)
        else:
            return (int((most_left + most_right) / 2), start)

    def check_multi_root(img, th=20, opt='up'):

        start = len(img) - th
        end = len(img)
        step = 1
        flag = False
        gray_img = copy.deepcopy(img)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        canvas = copy.deepcopy(img)
        x_arr = [0]
        y_arr = [0]

        if opt == 'up':
            start = th
            end = 0
            step = -1
        for col in range(start, end, step):
            #crop = gray_img[:col, :]
            if opt == 'up':
                crop = gray_img[end:col, :]
            else:
                crop = gray_img[col:end, :]
            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(crop)

            if cnt > 2:

                flag = True
                #continue

                for i in range(1, cnt):

                    (x, y, w, h, area) = stats[i]
                    if opt == 'up':
                        pt1 = get_root_pos(crop[y:y+h+1, x:x+w+1], th=5)
                        bias = 0

                    else:
                        pt1 = get_root_pos(crop[y:y+h, x:x+w], th=5, opt='lo')
                        bias = col

                    x_arr.append(pt1[0]+x)
                    y_arr.append(pt1[1]+y+bias)

                break
        if len(x_arr) == 1 or len(y_arr) == 1:
            return (0, 0), flag
        y_arr.remove(0)
        avg_x = int(sum(x_arr) / (len(x_arr)-1))
        avg_y = round(st.harmonic_mean(y_arr))

        if (avg_x, avg_y) == (0, 0):
            flag = False
        return (avg_x, avg_y), flag


    copied = copy.deepcopy(cropped_img)
    copied = cv2.cvtColor(copied, cv2.COLOR_BGR2GRAY)
    f = False
    pt1 = 0
    if num[0] == '1' or num[0] == '2': # upper
        # check multi root or not
        if num[1] in ['6', '7']:
            pt1, f = check_multi_root(cropped_img, 55)
        if not f :
            pt1 = get_root_pos(copied, 3)
        pt2 = get_crown_pos(copied, num, 5)
    else:                               # lower
        # check multi root or not
        if num[1] in ['6', '7']:
            pt1, f = check_multi_root(cropped_img, 55, 'lo')
        if not f :
            pt1 = get_root_pos(copied, 3, 'lo')
        pt2 = get_crown_pos(copied, num, 5, 'lo')
    #cv2.line(cropped_img, pt1, pt2, (255,0,0), 2)
    return pt1, pt2

def get_line_length(image):
    #if type(image) == int:
    #    print(image)
    first = [0, 0]
    last = [0, 0]
    flag = False
    img = copy.deepcopy(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for col in range(len(img)):
        for row in range(len(img[col])):
            if img[col][row] != 0:
                first = [row, col]
                flag = True
                break
        if flag:
            break

    flag = False
    for col in range(len(img)-1, 0, -1):
        for row in range(len(img[col])):
            if img[col][row] != 0:
                last = [row, col]
                flag = True
                break
        if flag:
            break

    a = first[0] - last[0]
    b = first[1] - last[1]
    l = math.sqrt( (a*a) + (b*b) )

    return l

def bonelevel_postprocessing(img, coord):
    most_right = 0
    most_left = 9999
    th = 40

    for c in coord[0]:
        x = c[0][0]
        y = c[0][1]
        if x > most_right:
            most_right = x
        if x < most_left:
            most_left = x

    h, w, c = img.shape
    mask = np.zeros([h, w], dtype=np.uint8)
    mask_post = np.zeros([h, w])
    most_left = int(most_left)
    most_right = int(most_right)
    most_left = [x for x in range(most_left, most_left+th)]
    most_right = [x for x in range(most_right, most_right-th, -1)]

    for c in coord[0]:
        x = round(c[0][0])
        y = round(c[0][1])
        if x in most_left:
            pass
        elif x in most_right:
            pass
        else:
            mask_post[y][x] = 255

    mask_post = mask_post.astype(np.uint8)
    mask_post = cv2.merge([mask_post, mask, mask])
    
    dst = copy.deepcopy(mask_post)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.dilate(dst, k)
    return dst

def get_bonelevel(img, res, all_):
    left_th = -10
    original_img = img.copy()

    bonelv = YOLO(BONELEVEL)
    bonelv_result = bonelv(img)
    
    if len(bonelv_result) == 0 :
        return img, {}, all_
    if bonelv_result[0].masks is None :
        return img, {}, all_
    cej = YOLO(CEJ)
    cej_result = cej(img)
    if len(cej_result) == 0 :
        return img, {}, all_
    bonelevel_dict = {}

    bonelevel_mask = np.zeros_like(original_img)
    cejimg_canvas = np.zeros_like(original_img)
    bonelevel_mask_cnt = get_squeeze_points(bonelv_result[0].masks.xy[0], copy.deepcopy(original_img), 0, cv2.CHAIN_APPROX_NONE)
    cv2.fillPoly(bonelevel_mask, bonelevel_mask_cnt, (255, 255, 255))
    cej_mask = np.zeros_like(original_img)
    try:
        for t in cej_result[0].masks.xy:
            cej_mask_cnt = get_squeeze_points(t, copy.deepcopy(original_img), 0, cv2.CHAIN_APPROX_NONE)
            cv2.fillPoly(cej_mask, cej_mask_cnt, (255, 255, 255))
            cejimg = bonelevel_postprocessing(copy.deepcopy(original_img), cej_mask_cnt)
            cejimg_canvas = cv2.addWeighted(cejimg_canvas, 1, cejimg, 1, 0)
            #all_ = cv2.addWeighted(all_, 1, cejimg, 1, 0)
    except:
        return img, {}, all_
    cejimg_canvas = cv2.cvtColor(cejimg_canvas, cv2.COLOR_BGR2RGB)
    
    bonelevelimg = bonelevel_postprocessing(copy.deepcopy(original_img), bonelevel_mask_cnt)

    for idx, coords in enumerate(res[0].masks.xy):
        try:
            # get real teeth number
            # keyval is teeth number
            keyval = res[0].names[res[0].boxes.cls[idx].item()]
            keyval = get_val(keyval)
            
            if keyval[1] == '8':
                continue
            #if keyval in bonelevel_dict.keys():
            #    continue
            # erode preprocessing and get bounding box
            res2 = get_squeeze_points(coords, copy.deepcopy(original_img), 1)
            mask = np.zeros_like(original_img)
            cv2.fillPoly(mask, res2, (255, 255, 255))

            cnt = res2[0]
            x, y, w, h = cv2.boundingRect(cnt)

            if keyval[1] == '8' and h <= w:
                continue

            pt1, pt2 = get_principal_axis(mask[y:y + h, x+left_th:x + w], keyval)
            cv2.line(original_img[y:y+h, x+left_th:x+w], pt1, pt2, (255, 0, 0), 1)  # (0, 255, 0) is the color of the line
            #cv2.line(all_[y:y+h, x+left_th:x+w], pt1, pt2, (255, 0, 0), 2)  # (0, 255, 0) is the color of the line
            a = pt1[0] - pt2[0]
            b = pt1[1] - pt2[1]
            teeth_len = math.sqrt((a * a) + (b * b))
            tooth_length_mask = np.zeros_like(original_img)
            cv2.line(tooth_length_mask[y:y + h, x + left_th:x + w], pt1, pt2, (255, 255, 255),
                     1)  # (0, 255, 0) is the color of the line
            periodontal_to_root = cv2.bitwise_and(tooth_length_mask, bonelevel_mask)
            periodontal_to_root_val = teeth_len - get_line_length(periodontal_to_root[y:y+h, x+left_th:x+w])

            tooth_length_mask = np.zeros_like(original_img)
            cv2.line(tooth_length_mask[y:y + h, x + left_th:x + w], pt1, pt2, (255, 255, 255),
                     1)  # (0, 255, 0) is the color of the line

            cej_to_root = cv2.bitwise_and(tooth_length_mask, cej_mask)
            cej_to_root_val = teeth_len - get_line_length(cej_to_root[y:y + h, x + left_th:x + w])

            PBL = periodontal_to_root_val/cej_to_root_val
            bonelevel_dict[keyval] = PBL*100

        except Exception as e:
            print(e)
            continue

    original_img = cv2.addWeighted(original_img, 1, bonelevelimg, 0.7, 0)
    all_ = cv2.addWeighted(all_, 1, bonelevelimg, 0.8, 0)
    original_img = cv2.addWeighted(original_img, 1, cejimg_canvas, 0.7, 0)
    final = cv2.vconcat([img, original_img])

    return final, bonelevel_dict, all_

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
    
def diagnosis_det(img, res, opt, all_):
    
    if opt == 'c' or opt =='caries':
        yolo_model = YOLO(CARIES)
        yolo_res = yolo_model(img)
    elif opt == 'p' or opt =='pl':
        yolo_model = YOLO(PERIAPICAL_LESION)
        yolo_res = yolo_model(img)
    # pl.predict(img, save=True)
    original_img = copy.deepcopy(img)

    final_result = []
    for boxes in yolo_res[0].boxes:
        for box in boxes:
            b = list(map(int, box.xyxy[0].tolist()))
            # x,y,w,h = i[0].item(), i[1].item(), i[2].item(), i[3].item()
            cv2.rectangle(original_img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
            if opt == 'c':
                cv2.rectangle(all_, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(all_, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            pl_bbox = [[b[0], b[1]], [b[2], b[3]]]
            # tooth seg 와 가장 크게 겹치는 점 찾기
            max_val = -1
            temp = -1
            for idx, coords in enumerate(res[0].masks.xy):
                # keyval is teeth number
                keyval = res[0].names[res[0].boxes.cls[idx].item()]
                keyval = get_val(keyval)

                res2 = get_squeeze_points(coords, copy.deepcopy(original_img), 0)
                cnt = res2[0]
                x, y, w, h = cv2.boundingRect(cnt)
                teeth_bbox = [[x,y], [x+w, y+h]]

                val, _ = get_iou(pl_bbox, teeth_bbox)
                if val > max_val:
                    max_val = val
                    temp = keyval
            final_result.append(temp)

    # original_img = cv2.addWeighted(original_img, 0.7, bonelevelimg, 0.5, 0)
    final = cv2.vconcat([img, original_img])
    return final, final_result, all_

def get_bonelevel_info(img, res):
    left_th = -10
    original_img = img.copy()

    bonelv = YOLO(BONELEVEL)
    bonelv_result = bonelv(img)
    
    if len(bonelv_result) == 0 :
        return {}
    if bonelv_result[0].masks is None :
        return {}
    cej = YOLO(CEJ)
    cej_result = cej(img)
    if len(cej_result) == 0 :
        return {}
    bonelevel_dict = {}

    bonelevel_mask = np.zeros_like(original_img)
    bonelevel_mask_cnt = get_squeeze_points(bonelv_result[0].masks.xy[0], copy.deepcopy(original_img), 0, cv2.CHAIN_APPROX_NONE)
    cv2.fillPoly(bonelevel_mask, bonelevel_mask_cnt, (255, 255, 255))
    cej_mask = np.zeros_like(original_img)
    try:
        for t in cej_result[0].masks.xy:
            cej_mask_cnt = get_squeeze_points(t, copy.deepcopy(original_img), 0, cv2.CHAIN_APPROX_NONE)
            cv2.fillPoly(cej_mask, cej_mask_cnt, (255, 255, 255))
            cejimg = bonelevel_postprocessing(copy.deepcopy(original_img), cej_mask_cnt)
            #all_ = cv2.addWeighted(all_, 1, cejimg, 1, 0)
    except:
        return {}
    
    bonelevelimg = bonelevel_postprocessing(copy.deepcopy(original_img), bonelevel_mask_cnt)

    for idx, coords in enumerate(res[0].masks.xy):
        try:
            keyval = res[0].names[res[0].boxes.cls[idx].item()]
            keyval = get_val(keyval)
            
            if keyval[1] == '8':
                continue

            res2 = get_squeeze_points(coords, copy.deepcopy(original_img), 1)
            mask = np.zeros_like(original_img)
            cv2.fillPoly(mask, res2, (255, 255, 255))

            cnt = res2[0]
            x, y, w, h = cv2.boundingRect(cnt)

            if keyval[1] == '8' and h <= w:
                continue

            pt1, pt2 = get_principal_axis(mask[y:y + h, x+left_th:x + w], keyval)
            cv2.line(original_img[y:y+h, x+left_th:x+w], pt1, pt2, (255, 0, 0), 1)  # (0, 255, 0) is the color of the line

            a = pt1[0] - pt2[0]
            b = pt1[1] - pt2[1]
            teeth_len = math.sqrt((a * a) + (b * b))
            tooth_length_mask = np.zeros_like(original_img)
            cv2.line(tooth_length_mask[y:y + h, x + left_th:x + w], pt1, pt2, (255, 255, 255),
                     1)  # (0, 255, 0) is the color of the line
            periodontal_to_root = cv2.bitwise_and(tooth_length_mask, bonelevel_mask)
            periodontal_to_root_val = teeth_len - get_line_length(periodontal_to_root[y:y+h, x+left_th:x+w])

            tooth_length_mask = np.zeros_like(original_img)
            cv2.line(tooth_length_mask[y:y + h, x + left_th:x + w], pt1, pt2, (255, 255, 255),
                     1)  # (0, 255, 0) is the color of the line

            cej_to_root = cv2.bitwise_and(tooth_length_mask, cej_mask)
            cej_to_root_val = teeth_len - get_line_length(cej_to_root[y:y + h, x + left_th:x + w])

            PBL = periodontal_to_root_val/cej_to_root_val
            bonelevel_dict[keyval] = PBL*100

        except Exception as e:
            print(e)
            continue

    return bonelevel_dict

def get_diagnosis_info(img, res, opt):
    
    if opt == 'c' or opt =='caries':
        yolo_model = YOLO(CARIES)
        yolo_res = yolo_model(img)
    elif opt == 'p' or opt =='pl':
        yolo_model = YOLO(PERIAPICAL_LESION)
        yolo_res = yolo_model(img)
    # pl.predict(img, save=True)
    original_img = copy.deepcopy(img)

    final_result = []
    for boxes in yolo_res[0].boxes:
        for box in boxes:
            b = list(map(int, box.xyxy[0].tolist()))
            # x,y,w,h = i[0].item(), i[1].item(), i[2].item(), i[3].item()

            pl_bbox = [[b[0], b[1]], [b[2], b[3]]]
            # tooth seg 와 가장 크게 겹치는 점 찾기
            max_val = -1
            temp = -1
            for idx, coords in enumerate(res[0].masks.xy):
                # keyval is teeth number
                keyval = res[0].names[res[0].boxes.cls[idx].item()]
                keyval = get_val(keyval)

                res2 = get_squeeze_points(coords, copy.deepcopy(original_img), 0)
                cnt = res2[0]
                x, y, w, h = cv2.boundingRect(cnt)
                teeth_bbox = [[x,y], [x+w, y+h]]

                val, _ = get_iou(pl_bbox, teeth_bbox)
                if val > max_val:
                    max_val = val
                    temp = keyval
            final_result.append(temp)

    # original_img = cv2.addWeighted(original_img, 0.7, bonelevelimg, 0.5, 0)

    return final_result


