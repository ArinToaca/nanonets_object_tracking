import copy
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

# Asta e DeepSort Trackerul
from deepsort import deepsort_rbc
from multitracker2 import JDETracker
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def draw_rect(image, xy1, xy2):
    color = (255, 0, 0)
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

def hungarian_matching(active_luggage_tracks, active_person_tracks, dist_thresh=175):
    # Luggage is from FairMOT and Person is from DeepSORT
    luggage_index = {}
    person_index = {}
    assig_matrix = np.matrix(np.ones((len(active_luggage_tracks), len(active_person_tracks))) * 9e3)

    for i, l_track in enumerate(active_luggage_tracks):
        luggage_index[i] = l_track.track_id
        luggage_center = l_track.tlwh

        for j, p_track in enumerate(active_person_tracks):
            person_index[j] = p_track.track_id
            person_center = p_track.to_tlwh()

            dist = np.linalg.norm(luggage_center - person_center)

            # Feel free to change this daca gasesti ceva mai bun
            if dist <= dist_thresh:
                assig_matrix[i, j] = dist

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
        luggage_to_person_mapping[row_id] = col_id

    return  luggage_to_person_mapping



class CounterProvider(object):
    def __init__(self):
        self.object_counter = 0

    def increment_and_get(self):
        val = self.object_counter
        self.object_counter += 1
        return val


OPT = namedtuple('OPT', 'conf_thres, track_buffer, nms_thres, min_box_area')
opt = OPT(0.2, 10, 0.4, 100)
counter_provider = CounterProvider()
tracker = JDETracker(opt, counter_provider=counter_provider)
deepsort_person = deepsort_rbc(wt_path='ckpts/model640.pt')

output_folder = sys.argv[4]

detections_pd = pd.read_csv(sys.argv[2])
cap = VideoCapture(sys.argv[1])
vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_list = []
dict_counter = {}
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
    dets = dets[dets['confidence'] > 0.4]
    luggage_dets = dets[(dets['name'] == 'backpack') | (
        dets['name'] == 'suitcase') | (dets['name'] == 'handbag')]
    person_dets = dets[(dets['name'] == 'person')]

    if luggage_dets.empty or person_dets.empty:
        continue

    online_luggages = tracker.update(
        luggage_dets[['xcenter', 'ycenter', 'width', 'height', 'confidence']].values.astype(float))

    people_scrs = person_dets['confidence'].values.astype(float)
    person_dets = deepsort_person.format_yolo_output(
        person_dets[['xcenter', 'ycenter', 'width', 'height']].values.astype(float))
    person_tracker , _ = deepsort_person.run_deep_sort(
                frame, people_scrs, person_dets)

    # next step: Hungarian assignment
    online_persons = [el for el in person_tracker.tracks if el.is_confirmed() or el.time_since_update > 1]
    l_to_p_mapping = hungarian_matching(online_luggages, online_persons)
    l_to_p_mapping_total.update(l_to_p_mapping) # keep person to luggage matching in mind for all frames

    for track in online_luggages:
        tid = track.track_id
        x, y, w, h = track.tlwh
        x1, y1, x2, y2 = xywh2xyxy(track.tlwh).astype(int)
        cropped_img = frame_for_cropping[y1:y2,x1:x2] # we only crop luggages
        if 0 in cropped_img.shape:
            continue
        if tid in dict_counter:
            dict_counter[tid] += 1
        else:
            dict_counter[tid] = 0
        p_id = str(l_to_p_mapping_total.get(tid, ''))
        n = dict_counter[tid]
        filename = f'p{p_id}_{tid}_{n}.jpg'
        p1 = int(x1), int(y1)
        p2 = int(x2), int(y2)
        p_c = int(x), int(y)
        frame = draw_rect(frame, p1, p2)
        frame = draw_id(frame, str(tid), p_c, color=(0, 0, 255))
        frame = draw_id(frame, filename, p1, fontScale=0.6)
        os.makedirs(os.path.join(output_folder, str(tid)), exist_ok=True)
        filename = os.path.join(output_folder, str(tid), filename)
        cv2.imwrite(filename, cropped_img)
    frame_list.append(frame)

write_video(sys.argv[3], frame_list, fps=25)
