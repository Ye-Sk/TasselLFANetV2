"""
@author: Jianxiong Ye
"""

import os
import cv2
import torch
import argparse
from pathlib import Path

from models.utils.traineval import check_dataset
from models.utils.model import annotator, scale_boxes, NMS, MltDetectionModel
from models.utils.helper import infer_mode, crement_path, print_info, logger, \
    Time_record, verify_img_size, LoadImages, LoadWebcam, colorstr


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/last.pt', help='model path')
    parser.add_argument('--source', type=str, default='data/RSAD/test/images', help='file/dir/URL/0(webcam)')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[608, 608], help='inference size h,w')
    parser.add_argument('--data', type=str, default='config/RSAD.yaml', help='config.yaml path')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--save-img', action='store_true', default=False, help='save images')
    opt = parser.parse_args()
    print_info(vars(opt))
    return opt

@infer_mode
def run(
        weights='weights/last.pt',
        source='data/images',
        imgsz=(608, 608),
        data=None,
        conf_thres=0.20,
        iou_thres=0.5,
        max_det=1000,
        save_img = True,
        color=(64, 64, 255),
        line_width=None,
):
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    if save_img:
        # Directories
        save_dir = crement_path('runs/infer/exp')  # increment run
        os.makedirs(save_dir, exist_ok=True)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MltDetectionModel(weights, device, data)
    stride, names = model.stride(), model.names
    imgsz = verify_img_size(imgsz, s=stride)

    dataset = LoadWebcam(source, img_size=imgsz, stride=stride) if webcam else LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    seen, IT = 0, (Time_record(), Time_record(), Time_record())
    for path, im, im0s, info in dataset:
        with IT[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        with IT[1]:  # Inference
            pred = model(im)

        with IT[2]:  # NMS
            pred = NMS(pred, conf_thres, iou_thres, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0 = (Path(path[i]), im0s[i].copy()) if webcam else (Path(path), im0s.copy())

            if save_img:
                save_path = f'{save_dir}/{p.name}'  # im.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                if save_img or webcam:   # Add bbox to image
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        annotator(im0, xyxy, label, color=color, line_width=line_width)

            if webcam:  # Stream results
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            elif save_img:  # Save results (image with detections)
                cv2.imwrite(save_path, im0)

        # Print time (inference-only)
        logger.info(f"{info}{p.name}, {len(det)} object, {IT[1].IT * 1E3:.1f}ms")

    # Print results
    logger.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % tuple(x.t / seen * 1E3 for x in IT))

    if save_img:
        logger.info(f"Image results saved to {colorstr('bold', save_dir)}")


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
