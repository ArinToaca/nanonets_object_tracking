import copy
import torchvision
import os
import sys
from collections import namedtuple

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from cv2 import VideoCapture
from scipy.optimize import linear_sum_assignment

from deepsort import deepsort_rbc
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


VIDEO_PATH, CSV_PATH, VIDEO_OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]

def draw_rect(image, xy1, xy2, color):
    thickness = 2
    return cv2.rectangle(image, xy1, xy2, color, thickness)


def draw_id(image, str_id, center, thickness=2, color=(255, 0, 0), fontScale=1):

    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(image, str_id, center, font,
                       fontScale, color, thickness, cv2.LINE_AA)


def write_video(file_path, frames, fps=25):
    print("Writing video")
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(frame)

    writer.release()

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y
def xywh2xyxy_multiple(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:,0] = x[:,0] - x[:,2] / 2  # top left x
    y[:,1] = x[:,1] - x[:,3] / 2  # top left y
    y[:,2] = x[:,0] + x[:,2] / 2  # bottom right x
    y[:,3] = x[:,1] + x[:,3] / 2  # bottom right y
    return y

def tlwh2xywh(tlwh):
    x1, y1, w, h = tlwh
    xc = x1 + w/2
    yc = y1 + h/2
    return xc, yc, w, h

def hungarian_matching(active_luggage_tracks, active_person_tracks, dist_thresh=0.65):
    # Luggage is from FairMOT and Person is from DeepSORT
    luggage_index = {}
    person_index = {}
    assig_matrix = np.matrix(np.ones((len(active_luggage_tracks), len(active_person_tracks))) * 9e3)
    ignore_pair = []
    for i, l_track in enumerate(active_luggage_tracks):
        luggage_index[i] = l_track.track_id
        luggage_center = tlwh2xywh(l_track.to_tlwh())
        lx, ly, lw, lh = luggage_center
        for j, p_track in enumerate(active_person_tracks):
            person_index[j] = p_track.track_id
            person_center = tlwh2xywh(p_track.to_tlwh())
            px, py, pw, ph = person_center
            #dist = np.linalg.norm(luggage_center - person_center)
            dist = abs(px - lx) / abs(lw + pw)
            dist += abs(py - ly) / abs(ph + lh)
            #print(f"DISTANCE IS {dist}")
            # Feel free to change this daca gasesti ceva mai bun
            if dist <= dist_thresh:
                assig_matrix[i, j] = dist
            else:
                ignore_pair.append((i,j))

    # Asta e practic hungarian assignment asta iti returneaza practic 2 liste 1 cu row indexes si
    # cealalta cu column indexes si atunci practic row_ind[i] vine assigned la col_ind[i]
    # E.g:
    # row_ind = [0, 3, 1, 4, 2]
    # col_ind = [1, 3, 2, 0, 4]
    # care inseamna ca luggage de pe pozitia 0 e assigned cu person de pe pozitia 1
    # luggage de pe pozitia 3 e assigned la person de pe pozitia 3 si tot asa

    row_ind, col_ind = linear_sum_assignment(assig_matrix)
    luggage_to_person_mapping = {}
    for row, col in zip(row_ind, col_ind):
        row_id = luggage_index[row]
        col_id = person_index[col]
        if (row, col) in ignore_pair:
            continue
        luggage_to_person_mapping[row_id] = col_id

    return  luggage_to_person_mapping


def preprocess_csv_vals(dets, yolo_preprocess_func):
    dets_scrs = dets['confidence'].values.astype(float)
    dets_vals  = dets[['xcenter', 'ycenter', 'width', 'height']].values.astype(float)
    keep_idx_nms = torchvision.ops.nms(torch.tensor(xywh2xyxy_multiple(dets_vals)),torch.tensor(dets_scrs), 0.5)
    dets_vals = dets_vals[keep_idx_nms].reshape(-1,4)
    dets_scrs = dets_scrs[keep_idx_nms].reshape(-1)
    dets_vals = yolo_preprocess_func(dets_vals)
    return dets_scrs, dets_vals

def draw_tracks(online_tracks, frame, color=(0,0,255)):
    for track in online_tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        x1, y1, w, h = track.to_tlwh()
        x2 = x1 + w
        y2 = y1 + h
        if w <= 0 or h <= 0:
            continue
        p1 = int(x1), int(y1)
        p2 = int(x2), int(y2)
        p_c = int(x1 + w/2), int(y1 + h/2)
        frame = draw_rect(frame, p1, p2, color)
        frame = draw_id(frame, str(tid), p_c, color=color)
    return frame
    
class CounterProvider(object):
    def __init__(self):
        self.object_counter = 0

    def increment_and_get(self):
        val = self.object_counter
        self.object_counter += 1
        return val

ORANGE_COLOR = (0, 165, 255)
OPT = namedtuple('OPT', 'conf_thres, track_buffer, nms_thres, min_box_area')
opt = OPT(0.2, 10, 0.4, 100)
counter_provider = CounterProvider()
deepsort_luggage = deepsort_rbc(wt_path='ckpts/model80.pt')
deepsort_person = deepsort_rbc(wt_path='ckpts/old/model80.pt')

detections_pd = pd.read_csv(CSV_PATH)
cap = VideoCapture(VIDEO_PATH)
vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_list = []
dict_counter = {}
abandoned_by = {}
abandoned_coords = {}
truly_abandoned_by = {}
consecutive_abandoned = {} # 'id' -> frames obj is stationary
suspicious_recovered_by = {}
print(
    f"Processing images from video {sys.argv[1]} with detections from {sys.argv[2]}")
l_to_p_mapping_total = {}
for count in tqdm.tqdm(range(vid_length)):
    ret, frame = cap.read()
    frame_for_cropping = copy.deepcopy(frame)
    if frame is None:
        break
    if 0 in frame.shape:
        continue

    # from CSV file, we take persons and luggage yolov5 detections
    dets = detections_pd[detections_pd['frame_id'] == count]
    dets = dets[dets['confidence'] > 0.6]

    luggage_dets = dets[(dets['name'] == 'backpack') | (
        dets['name'] == 'suitcase') | (dets['name'] == 'handbag')]
    
    person_dets = dets[(dets['name'] == 'person')]

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if luggage_dets.empty:
        # potential for visual tracking from previous frame
        online_luggage_tracks = deepsort_luggage.run_deep_sort(img_rgb, [], [])
    else:
        luggage_scrs, luggage_dets_vals = preprocess_csv_vals(luggage_dets, deepsort_luggage.format_yolo_output)
        online_luggage, _ = deepsort_luggage.run_deep_sort(
                    img_rgb, luggage_scrs, luggage_dets_vals)
        online_luggage_tracks = online_luggage.tracks

    if person_dets.empty:
        online_person_tracks = deepsort_person.run_deep_sort(img_rgb, [], [])
    else:
        person_scrs, person_dets_vals = preprocess_csv_vals(person_dets, deepsort_person.format_yolo_output)
        online_persons, _ = deepsort_person.run_deep_sort(img_rgb, person_scrs, person_dets_vals)
        online_person_tracks = online_persons.tracks

    DIST_THRESH=1 # distance scaled by size, [0,1], smaller should mean more strict
    MOTION_DIST_THRESH =  5 #motion threshold for object to be considered stationary
    MOTION_STOP_FRAMES = 15 # how many frames the abandoned object stopped moving
    # two to one mapping maximum, we assume people can hold a maximum of two baggages
    l_to_p_map = hungarian_matching(online_luggage_tracks, online_person_tracks, dist_thresh=DIST_THRESH)
    unmatched_luggage_tracks = [track for track in online_luggage_tracks if track.track_id not in l_to_p_map.keys()]
    l_to_p_map2 = hungarian_matching(unmatched_luggage_tracks, online_person_tracks, dist_thresh=DIST_THRESH)
    l_to_p_map.update(l_to_p_map2)

    for luggage_track in online_luggage_tracks:
        l_id = luggage_track.track_id 
        if l_id not in l_to_p_map:
            if l_id not in abandoned_by:
                abandoned_by[l_id] = str(l_to_p_mapping_total.get(l_id, 'unknown'))
                abandoned_coords[l_id] = tlwh2xywh(luggage_track.to_tlwh())
            else:
                prev_coords = abandoned_coords[l_id]
                current_coords = tlwh2xywh(luggage_track.to_tlwh())
                px, py, _, _ = prev_coords
                cx, cy, _, _ = current_coords
                if abs(px - cx) + abs(py - cy) <= MOTION_DIST_THRESH:
                    consecutive_abandoned[l_id] = consecutive_abandoned.get(l_id, 0) + 1
                else:
                    consecutive_abandoned[l_id] = 0   
        elif l_id in l_to_p_map and l_id in truly_abandoned_by.keys():
            # recovered baggage, trigger matching of baggage with person
            abandoned_by_p = truly_abandoned_by[l_id] # person id
            recovered_by_p = l_to_p_map[l_id]
            if str(abandoned_by) != (recovered_by_p): # we can use siamese_net here
                suspicious_recovered_by[l_id] = recovered_by_p
            truly_abandoned_by.pop(l_id)
            consecutive_abandoned.pop(l_id, 0)

        if consecutive_abandoned.get(l_id, 0) >= MOTION_STOP_FRAMES:
            truly_abandoned_by[l_id] = str(l_to_p_mapping_total.get(l_id, 'unknown'))
            consecutive_abandoned.pop(l_id, 0)
    
    l_to_p_mapping_total.update(l_to_p_map)

    frame = draw_tracks(online_person_tracks, frame, color=(0,255,0))
    frame = draw_tracks(online_luggage_tracks, frame, color=(0,255,0))

    suspicious_people = [track for track in online_person_tracks if str(track.track_id) in suspicious_recovered_by.values()]
    suspicious_luggages = [track for track in online_luggage_tracks if track.track_id in suspicious_recovered_by.keys()]
    frame = draw_tracks(suspicious_people, frame, color=ORANGE_COLOR)
    frame = draw_tracks(suspicious_luggages, frame, color=ORANGE_COLOR)

    abandoned_people = [track for track in online_person_tracks if str(track.track_id) in truly_abandoned_by.values()]
    abandoned_luggages = [track for track in online_luggage_tracks if track.track_id in truly_abandoned_by.keys()]
    frame = draw_tracks(abandoned_people, frame, color=(0,0,255))
    frame = draw_tracks(abandoned_luggages, frame, color=(0,0,255))

    # FOR DEBUGGING
    # ok_luggage_tracks = [track for track in online_luggage_tracks if track.track_id not in truly_abandoned_by or track.track_id not in suspicious_recovered_by]
    # ok_person_tracks = [track for track in online_person_tracks if track.track_id not in truly_abandoned_by.values() or track.track_id not in suspicious_recovered_by.values()]

    # frame = draw_tracks(ok_luggage_tracks, frame, color=(0,255,0))
    # frame = draw_tracks(ok_person_tracks, frame, color=(0,255,0))
    frame_list.append(frame)

write_video(VIDEO_OUT_PATH, frame_list, fps=25)