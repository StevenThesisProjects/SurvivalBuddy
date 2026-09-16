import json
import numpy as np

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_stiou(gt_boxes_dict, pred_boxes_dict):
    gt_frames = set(gt_boxes_dict.keys())
    pred_frames = set(pred_boxes_dict.keys())

    # Định nghĩa tập không gian và thời gian 
    intersection_frames = gt_frames.intersection(pred_frames)   # Tập intersection 
    union_frames = gt_frames.union(pred_frames)                 # Tập union 

    if not union_frames:
        return 0.0

    total_iou = 0.0
    for f in intersection_frames:
        total_iou += calculate_iou(gt_boxes_dict[f], pred_boxes_dict[f])

    # Công thức STIoU 
    stiou = total_iou / len(union_frames)
    return stiou

def evaluate_submission(gt_json_path, pred_json_path):
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)

    gt_dict = {}
    for item in gt_data:
        v_id = item['video_id']
        bboxes = item['annotations'][0]['bboxes'] if 'annotations' in item else item['detections'][0]['bboxes']
        gt_dict[v_id] = bboxes

    pred_dict = {}
    for item in pred_data:
        if 'detections' in item:
            pred_dict[item['video_id']] = item['detections'][0]['bboxes']
        elif 'annotations' in item:
            pred_dict[item['video_id']] = item['annotations'][0]['bboxes']

    all_video_ids = set(gt_dict.keys())
    stiou_scores = []

    for video_id in all_video_ids:
        gt_boxes = {b['frame']: [b['x1'], b['y1'], b['x2'], b['y2']] for b in gt_dict.get(video_id, [])}
        pred_boxes = {b['frame']: [b['x1'], b['y1'], b['x2'], b['y2']] for b in pred_dict.get(video_id, [])}

        video_stiou = calculate_stiou(gt_boxes, pred_boxes)
        stiou_scores.append(video_stiou)

    # Final Score 
    final_score = np.mean(stiou_scores) if stiou_scores else 0.0
    return final_score

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        gt_path = sys.argv[1]
        pred_path = sys.argv[2]
        score = evaluate_submission(gt_path, pred_path)
        print(f"Kết quả đánh giá hệ thống - Final STIoU Score: {score:.4f}")
    else:
        print("STIoU Metric is ready to run.")