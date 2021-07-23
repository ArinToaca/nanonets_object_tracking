import enum
import cv2
import torch
import warnings
import numpy as np
from deepsort import deepsort_rbc

from scipy.optimize import linear_sum_assignment


warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_gt(image, frame_id, gt_dict):

    if frame_id not in gt_dict.keys() or gt_dict[frame_id] == []:
        return None, None, None

    frame_info = gt_dict[frame_id]

    detections = []
    ids = []
    out_scores = []
    classes = []

    for i in range(len(frame_info)):

        coords = frame_info[i]['coords']

        x1, y1, w, h = coords
        x2 = x1 + w
        y2 = y1 + h

        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)

        detections.append([x1, y1, w, h])
        out_scores.append(frame_info[i]['conf'])
        classes.append(frame_info[i]['class'])

    return detections, out_scores, classes


def get_dict(filename, separator: str = ','):
    with open(filename) as f:
        d = f.readlines()

    d = list(map(lambda x: x.strip(), d))

    last_frame = int(d[-1].split(separator)[0])

    gt_dict = {x: [] for x in range(last_frame + 1)}

    for i in range(len(d)):
        a = list(d[i].split(separator))
        a = list(map(float, a))

        coords = a[2:6]
        confidence = a[6]
        obj_class = a[1]

        gt_dict[a[0]].append(
            {'coords': coords, 'conf': confidence, 'class': obj_class})

    return gt_dict


def get_mask(filename):
    mask = cv2.imread(filename, 0)
    mask = mask / 255.0
    return mask


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y


if __name__ == '__main__':

    # Load detections for the video. Options available: yolo,ssd and mask-rcnn
    filename = '/data_disk/ds_ro/amuresan/yolov5/runs/detect/exp5/labels/AVSS_all-detections.txt'
    gt_dict = get_dict(filename, separator=' ')

    # Load the video here
    cap = cv2.VideoCapture(
        '/data_disk/ds_ro/amuresan/yolov5/AVSS_AB_Easy_Divx.avi')
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    # An optional mask for the given video, to focus on the road.
    mask = get_mask('roi.jpg')

    # Initialize deep sort.
    deepsort_person = deepsort_rbc(wt_path='ckpts/model80.pt')
    deepsort_luggage = deepsort_rbc(wt_path='ckpts/model80.pt')

    frame_id = 1

    mask = np.expand_dims(mask, 2)
    mask = np.repeat(mask, 3, 2)

    mask = None

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test_hungarian_assig.avi',
                          fourcc, fps, (int(width), int(height)))

    while True:
        print(frame_id)

        ret, frame = cap.read()
        if ret is False:
            frame_id += 1
            break

        if mask:
            frame = frame * mask

        frame = frame.astype(np.uint8)

        # detections, out_scores = get_gt(frame, frame_id, gt_dict)
        dets_scores_classes = get_gt(frame, frame_id, gt_dict)

        if len(dets_scores_classes) == 3:
            detections = dets_scores_classes[0]
            out_scores = dets_scores_classes[1]
            classes = dets_scores_classes[2]
        else:
            frame_id += 1
            continue

        if detections is None:
            print("No dets")
            frame_id += 1
            continue

        people_dets = []
        people_scrs = []

        luggage_dets = []
        luggage_scrs = []

        for det, scr, cls in zip(detections, out_scores, classes):
            cls = int(cls)

            # check if the class is person
            if cls == 0:
                people_dets.append(det)
                people_scrs.append(scr)

            # check if the class is backpack, suitcase or handbag
            if cls in [26, 28, 30, 32]:
                luggage_dets.append(det)
                luggage_scrs.append(scr)

        people_dets = np.array(people_dets)
        people_scrs = np.array(people_scrs)

        luggage_dets = np.array(luggage_dets)
        luggage_scrs = np.array(luggage_scrs)

        scaler = np.array([frame.shape[1], frame.shape[0],
                          frame.shape[1], frame.shape[0]])

        person_index_dict = {}
        luggage_index_dict = {}

        if len(people_dets) > 0:
            people_dets = people_dets * scaler
            person_tracker, person_detections_class = deepsort_person.run_deep_sort(
                frame, people_scrs, people_dets)

            for track in person_tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                # Get the corrected/predicted bounding box
                bbox = xywh2xyxy(track.to_tlwh())
                # Get the ID for the particular track.
                id_num = str(track.track_id)
                # Get the feature vector corresponding to the detection.
                features = track.features

                # Draw bbox from tracker.
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
                    bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, str(id_num), (int(bbox[0]), int(
                    bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
                cv2.circle(frame, (int((bbox[0] + bbox[2]) / 2),
                           int((bbox[1] + bbox[3]) / 2)), radius=5,
                           color=(0, 0, 255), thickness=-1)

                # Draw bbox from detector. Just to compare.
                for det in person_detections_class:
                    bbox = xywh2xyxy(det.tlwh)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
                        bbox[2]), int(bbox[3])), (255, 255, 0), 2)

        if len(luggage_dets) > 0:
            luggage_dets = luggage_dets * scaler
            luggage_tracker, luggage_detections_class = deepsort_luggage.run_deep_sort(
                frame, luggage_scrs, luggage_dets)

            for track in luggage_tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                # Get the corrected/predicted bounding box
                bbox = xywh2xyxy(track.to_tlwh())
                # Get the ID for the particular track.
                id_num = str(track.track_id)
                # Get the feature vector corresponding to the detection.
                features = track.features

                # Draw bbox from tracker.
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
                    bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, str(id_num), (int(bbox[0]), int(
                    bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
                cv2.circle(frame, (int((bbox[0] + bbox[2]) / 2),
                           int((bbox[1] + bbox[3]) / 2)), radius=5,
                           color=(0, 0, 255), thickness=-1)

                # Draw bbox from detector. Just to compare.
                for det in luggage_detections_class:
                    bbox = xywh2xyxy(det.tlwh)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
                        bbox[2]), int(bbox[3])), (255, 255, 0), 2)

        active_luggage_tracks = []
        active_person_tracks = []

        if len(luggage_dets) > 0:
            for l_track in luggage_tracker.tracks:
                if not l_track.is_confirmed() or l_track.time_since_update > 1:
                    continue
                else:
                    active_luggage_tracks.append(l_track)

        if len(people_dets) > 0:
            for p_track in person_tracker.tracks:
                if not p_track.is_confirmed() or p_track.time_since_update > 1:
                    continue
                else:
                    active_person_tracks.append(p_track)

        # assig_matrix = np.zeros((len(luggage_tracker.tracks), len(person_tracker.tracks)))
        assig_matrix = np.matrix(np.ones((len(active_luggage_tracks), len(active_person_tracks))) * 9e3)

        for i, l_track in enumerate(active_luggage_tracks):
            luggage_index_dict[i] = l_track.track_id
            l_bbox = xywh2xyxy(l_track.to_tlwh())

            if len(active_person_tracks) > 0:
                luggage_center = np.array(
                    [int((l_bbox[0] + l_bbox[2]) / 2),
                        int((l_bbox[1] + l_bbox[3]) / 2)])

                for j, p_track in enumerate(active_person_tracks):
                    person_index_dict[j] = p_track.track_id
                    p_bbox = xywh2xyxy(p_track.to_tlwh())

                    person_center = np.array(
                        [int((p_bbox[0] + p_bbox[2]) / 2),
                            int((p_bbox[1] + p_bbox[3]) / 2)])

                    dist = np.linalg.norm(luggage_center - person_center)

                    if dist <= 175:
                        assig_matrix[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(assig_matrix)

        if not luggage_index_dict:
            frame_id += 1
            out.write(frame)
        else:
            offset = 50
            for row, col in zip(row_ind, col_ind):
                if assig_matrix[row, col] <= 175:
                    txt = f"Luggage {luggage_index_dict[row]} -> Person {person_index_dict[col]}: {assig_matrix[row, col]:.2f}"
                    cv2.putText(frame, txt, (30, offset), 0, 5e-3 * 100, (0, 255, 0), 2)
                    offset += 25

            out.write(frame)
            frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
