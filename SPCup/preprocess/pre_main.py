# coding:utf-8
import argparse
import numpy as np
import os
import cv2


def main(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    class_names = os.listdir(args.data_dir)
    for class_name in class_names:
        img_names = os.listdir(os.path.join(args.data_dir, class_name))
        class_dir = os.path.join(args.out_dir, class_name)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        for img_name in img_names:
            full_path = os.path.join(args.data_dir, class_name, img_name)
            out_path = os.path.join(args.out_dir, class_name, img_name)
            img = cv2.imread(full_path)
            if args.compression:
                cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                print(out_path + ' quality: ' + str(args.quality) + '%')
            elif args.cropping:
                pass
            elif args.contrast_enhancement:
                # 转到YCbCr，并在Y通道做直方图均衡化
                imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
                equ = cv2.equalizeHist(imgYCC[:, :, 0])
                imgYCC[:, :, 0] = equ
                img = cv2.cvtColor(imgYCC, cv2.COLOR_YCR_CB2BGR)
                # avg = np.mean(img)
                # img = avg + (img - avg) * (1 + args.percent)
                cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                print(out_path + ' quality: ' + str(args.quality) + '%')
            else:
                raise RuntimeError('Please chose a pre-process type!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process')
    parser.add_argument('-d', '--data_dir', help='Data direction', type=str, default='../../data')
    parser.add_argument('-o', '--out_dir', help='Output direction', type=str)
    parser.add_argument('-q', '--quality', help='JPEG image quality', type=int, default=100)
    parser.add_argument('-p', '--percent', help='contrast enhancement p% (-1<=p<=1)', type=float, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-com', '--compression', help='JPEG re-compression', action="store_true")
    group.add_argument('-cro', '--cropping', help='Cropping', action="store_true")
    group.add_argument('-con', '--contrast_enhancement', help='Contrast enhancement', action="store_true")
    args = parser.parse_args()
    main(args)
