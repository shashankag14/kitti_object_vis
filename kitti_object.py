""" Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import os
import sys
import numpy as np
import cv2.cv2 as cv2
import kitti_util as utils
from kitti_util import Object3d
import argparse
import tqdm
from pathlib import Path
from bounding_box import bounding_box as bb
import logging
import mayavi.mlab as mlab
from viz_util import draw_lidar, draw_gt_boxes3d
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mayavi"))

max_color = 30
cbox = np.array([[0, 70.4], [-40, 40], [-3, 1]])
colors = utils.random_colors(max_color)


class KITTIDataset(object):
    """Load and parse object data into a usable format."""

    def __init__(self, args):
        """root_dir contains training and testing folders"""
        self.root_dir = args.dir
        self.split = args.split
        self.split_dir = os.path.join(self.root_dir, self.split)
        self.args = args

        lidar_dir_name = "velodyne"
        image_dir_name = "image_02" if args.tracking else "image_2"
        label_dir_name = "label_02" if args.tracking else "label_2"
        calib_dir_name = "calib"
        pred_dir_name = "pred"
        depth_dir_name = "depth"

        self.image_dir = args.image_dir if args.image_dir else os.path.join(self.split_dir, image_dir_name)
        self.label_dir = args.label_dir if args.label_dir else os.path.join(self.split_dir, label_dir_name)
        self.lidar_dir = args.lidar_dir if args.lidar_dir else os.path.join(self.split_dir, lidar_dir_name)
        self.calib_dir = args.calib_dir if args.calib_dir else os.path.join(self.split_dir, calib_dir_name)
        self.depth_dir = args.depth_dir if args.depth_dir else os.path.join(self.split_dir, depth_dir_name)
        self.pred_dir = args.pred_dir if args.pred_dir else os.path.join(self.split_dir, pred_dir_name)

        if args.tracking:
            # load seq ids and count total num. of sample
            self.seq_ids = args.seq_ids if args.seq_ids else os.listdir(self.image_dir)
            self.seq_ids = sorted(["{:04d}".format(int(seq_id)) for seq_id in self.seq_ids], key=lambda x: int(x))
            print("Num. of {} sequences: {}".format(self.split, len(self.seq_ids)))

            self.sample_ids = dict()
            self.num_samples = dict()
            self._trk_label_all_seqs = dict()
            self._trk_pred_all_seqs = dict()
            for seq_id in self.seq_ids:
                seq_img_paths = os.path.join(self.image_dir, seq_id)
                self.sample_ids[seq_id] = sorted([int(sample.split(".")[0]) for sample in os.listdir(seq_img_paths)])
                self.num_samples[seq_id] = len(self.sample_ids[seq_id])

                if args.split == "training":
                    seq_labels_path = os.path.join(self.label_dir, seq_id + ".txt")
                    trk_label_all_frames = defaultdict(list)
                    for line in open(seq_labels_path, 'r'):
                        line = line.strip()
                        tokens = line.split(" ")
                        tokens[3:] = [float(x) for x in tokens[3:]]
                        frame_id = int(tokens[0])
                        track_id = tokens[1]
                        data = tokens[3:] + ["-1"] + [track_id]  # -1 because labels do not have confidence
                        trk_label_all_frames[frame_id].append(Object3d(data=data))
                    self._trk_label_all_seqs[seq_id] = trk_label_all_frames

                if args.show_preds:
                    seq_preds_path = os.path.join(self.pred_dir, seq_id + ".txt")
                    trk_preds_all_frames = defaultdict(list)
                    for line in open(seq_preds_path, 'r'):
                        line = line.strip()
                        tokens = line.split(" ")
                        tokens[3:] = [float(x) for x in tokens[3:]]
                        frame_id = int(tokens[0])
                        track_id = tokens[1]
                        data = tokens[2:] + [track_id]
                        trk_preds_all_frames[frame_id].append(Object3d(data=data))
                    self._trk_pred_all_seqs[seq_id] = trk_preds_all_frames
        else:
            self.sample_ids = sorted([int(sample.split(".")[0]) for sample in os.listdir(self.image_dir)])
            self.num_samples = len(self.sample_ids)

        if args.inds_file_path:
            with open(args.inds_file_path, 'r') as inds_file:
                self.sample_ids = [int(line) for line in inds_file.readlines()]
        elif args.ind is not None:
            self.sample_ids = [int(args.ind)]

        print("Total num. of {} samples: {}".format(self.split, len(self)))

    def __len__(self):
        if isinstance(self.num_samples, dict):
            total_num_samples = 0
            for _, num_sample in self.num_samples.items():
                total_num_samples += num_sample
            return total_num_samples
        else:
            return self.num_samples

    def __iter__(self):
        if isinstance(self.sample_ids, dict):
            for seq_id, sample_ids in self.sample_ids.items():
                for sample_id in sample_ids:
                    yield sample_id, seq_id
        else:
            for sample_id in self.sample_ids:
                yield sample_id

    def check_idx(self, idx, seq_idx=None):
        if isinstance(self.num_samples, dict) and seq_idx is not None:
            num_samples = self.num_samples[seq_idx]
        else:
            num_samples = self.num_samples
        assert idx < num_samples

    def get_image(self, idx, seq_idx=None):
        self.check_idx(idx, seq_idx)
        if seq_idx:
            img_path = os.path.join(self.image_dir, "{:04d}".format(int(seq_idx)), "{:06d}.png".format(int(idx)))
        else:
            img_path = os.path.join(self.image_dir, "{:06d}.png".format(int(idx)))
        return utils.load_image(img_path)

    def get_lidar(self, idx, seq_idx=None, dtype=np.float32, n_vec=4):
        self.check_idx(idx, seq_idx)
        if seq_idx:
            lidar_path = os.path.join(self.lidar_dir, "{:04d}".format(int(seq_idx)), "{:06d}.bin".format(int(idx)))
        else:
            lidar_path = os.path.join(self.lidar_dir, "{:06d}.png".format(int(idx)))
        return utils.load_velo_scan(lidar_path, dtype, n_vec)

    def get_calibration(self, idx, seq_idx=None):
        self.check_idx(idx, seq_idx)
        if seq_idx:
            calib_path = os.path.join(self.calib_dir, "{:04d}.txt".format(int(seq_idx)))
        else:
            calib_path = os.path.join(self.calib_dir, "{:06d}.txt".format(int(idx)))
        return utils.Calibration(calib_path)

    def get_label_objects(self, idx, seq_idx=None):
        self.check_idx(idx, seq_idx)
        if seq_idx:
            return self._trk_label_all_seqs[seq_idx][idx]
        else:
            label_path = os.path.join(self.label_dir, "{:06d}.txt".format(int(idx)))
            return utils.read_label(label_path)

    def get_pred_objets(self, idx, seq_idx=None):
        self.check_idx(idx, seq_idx)
        if seq_idx:
            return self._trk_pred_all_seqs[seq_idx][idx]
        else:
            label_path = os.path.join(self.label_dir, "{:06d}.txt".format(int(idx)))
            return utils.read_label(label_path)

    def get_depth(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % idx)
        return utils.load_depth(img_filename)


class Visualizer(object):
    def __init__(self, args):
        self.dataset = KITTIDataset(args)

        self.args = args
        self.fig = None
        self.video_writer = None
        self.output_path = None
        self.camera_output = None
        self.lidar_output = None

        if args.demo:
            mlab.options.offscreen = True

            if not os.path.exists("demo"):
                os.mkdir('demo')
            if 'lidar' in args.demo and 'camera' in args.demo:
                assert 'lidar' in args.save and 'camera' in args.save
                self.video_writer = cv2.VideoWriter('demo/lidar_camera.avi',
                                                    cv2.VideoWriter_fourcc('M', 'J',  'P', 'G'),
                                                    20, (1248, 384 * 2))
            elif 'lidar' in args.demo:
                assert 'lidar' in args.save
                self.video_writer = cv2.VideoWriter(f'demo/{args.demo[0]}.avi',
                                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                                    20, (1248, 384))
            else:
                self.video_writer = cv2.VideoWriter(f'demo/{args.demo[0]}.avi',
                                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                                    20, (1248, 384))
        if args.save:
            mlab.options.offscreen = True

            self.output_path = Path('output')
            self.output_path.mkdir(exist_ok=True)
            if 'camera' in args.save:
                self.camera_output = self.output_path / 'image_02'
                self.camera_output.mkdir(exist_ok=True)
            if 'lidar' in args.save:
                self.lidar_output = self.output_path / 'velodyne'
                self.lidar_output.mkdir(exist_ok=True)

        if args.show_lidar_with_depth:
            self.fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None,
                                   engine=None, size=(int(2 * 634), int(2 * 477)))

    def run(self):
        # TODO(farzad) multithreading has a problem with mayavi.
        if args.workers > 0:
            import concurrent.futures as futures
            with futures.ThreadPoolExecutor(args.workers) as executor:
                executor.map(vis.run_pipeline, self.dataset)
        else:
            for data_index in tqdm.tqdm(self.dataset):
                if isinstance(data_index, tuple):
                    self.run_pipeline(*data_index)
                else:
                    self.run_pipeline(data_index)

    def show_image_with_boxes(self, img, objects, calib, preds=None, show3d=True, score_threshold=0.60):
        """ Show image with 2D bounding boxes """
        img = np.copy(img)

        for obj in objects:
            if obj.type == "DontCare":
                continue
            if show3d:
                # for predictions
                if hasattr(obj, 'id'):
                    color = tuple([int(c * 255) for c in colors[int(obj.id) % max_color]])
                else:
                    color = (0, 255, 0)
                box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
                if box3d_pts_2d is None:
                    continue
                img = utils.draw_projected_box3d(img, box3d_pts_2d, color=color)
            else:
                # for predictions
                if hasattr(obj, 'id'):
                    color = None
                    label = str(obj.type)[:3] + ' %d' % obj.id
                else:
                    color = 'green'
                    label = None
                pos = int(obj.xmin), int(obj.ymin), int(obj.xmax), int(obj.ymax)
                bb.add(img, *pos, color=color, label=label)
        if preds is not None:
            for pred in preds:
                if pred.type == "DontCare":
                    continue
                if hasattr(pred, 'score') and pred.score < score_threshold:
                    continue
                if show3d:
                    # for predictions
                    if hasattr(pred, 'id'):
                        color = tuple([int(c * 255) for c in colors[int(pred.id) % max_color]])
                    else:
                        color = (0, 255, 0)
                    box3d_pts_2d, _ = utils.compute_box_3d(pred, calib.P)
                    if box3d_pts_2d is None:
                        continue
                    img = utils.draw_projected_box3d(img, box3d_pts_2d, color=color)
                else:
                    # for predictions
                    if hasattr(pred, 'id'):
                        color = None
                        label = str(pred.type)[:3] + ' %d' % pred.id
                    else:
                        color = 'blue'
                        label = None
                    pos = int(pred.xmin), int(pred.ymin), int(pred.xmax), int(pred.ymax)
                    bb.add(img, *pos, color=color, label=label)
        return img

    def show_image_with_boxes_3type(self, img, objects, objects2d, name, objects_pred):
        """ Show image with 2D bounding boxes """
        img1 = np.copy(img)  # for 2d bbox
        type_list = ["Pedestrian", "Car", "Cyclist"]
        # draw Label
        color = (0, 255, 0)
        for obj in objects:
            if obj.type not in type_list:
                continue
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                color,
                3,
            )
        startx = 5
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_lables = [obj.type for obj in objects if obj.type in type_list]
        text_lables.insert(0, "Label:")
        for n in range(len(text_lables)):
            text_pos = (startx, 25 * (n + 1))
            cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
        # draw 2D Pred
        color = (0, 0, 255)
        for obj in objects2d:
            cv2.rectangle(
                img1,
                (int(obj.box2d[0]), int(obj.box2d[1])),
                (int(obj.box2d[2]), int(obj.box2d[3])),
                color,
                2,
            )
        startx = 85
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_lables = [type_list[obj.typeid - 1] for obj in objects2d]
        text_lables.insert(0, "2D Pred:")
        for n in range(len(text_lables)):
            text_pos = (startx, 25 * (n + 1))
            cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
        # draw 3D Pred
        if objects_pred is not None:
            color = (255, 0, 0)
            for obj in objects_pred:
                if obj.type not in type_list:
                    continue
                cv2.rectangle(
                    img1,
                    (int(obj.xmin), int(obj.ymin)),
                    (int(obj.xmax), int(obj.ymax)),
                    color,
                    1,
                )
            startx = 165
            font = cv2.FONT_HERSHEY_SIMPLEX

            text_lables = [obj.type for obj in objects_pred if obj.type in type_list]
            text_lables.insert(0, "3D Pred:")
            for n in range(len(text_lables)):
                text_pos = (startx, 25 * (n + 1))
                cv2.putText(
                    img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA
                )

        cv2.imshow("with_bbox", img1)
        cv2.imwrite("imgs/" + str(name) + ".png", img1)

    def show_lidar_with_depth(self, pc_velo, objects, calib, img_fov=False, img_width=None, img_height=None,
                              objects_pred=None, depth=None, constraint_box=False, pc_label=False, save=False,
                              score_threshold=0.6):

        """ Show all LiDAR points.
            Draw 3d box in LiDAR point cloud (in velo coord system) """

        print(("All point num: ", pc_velo.shape[0]))
        if img_fov:
            pc_velo_index = utils.get_lidar_index_in_image_fov(
                pc_velo[:, :3], calib, 0, 0, img_width, img_height
            )
            pc_velo = pc_velo[pc_velo_index, :]
            print(("FOV point num: ", pc_velo.shape))
        print("pc_velo", pc_velo.shape)
        draw_lidar(pc_velo, fig=self.fig, pc_label=pc_label)

        # Draw depth
        if depth is not None:
            depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

            indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
            depth_pc_velo = np.hstack((depth_pc_velo, indensity))
            print("depth_pc_velo:", depth_pc_velo.shape)
            print("depth_pc_velo:", type(depth_pc_velo))
            print(depth_pc_velo[:5])
            draw_lidar(depth_pc_velo, fig=self.fig)

            if save:
                data_idx = 0
                vely_dir = "data/object/training/depth_pc"
                save_filename = os.path.join(vely_dir, "%06d.bin" % data_idx)
                print(save_filename)
                # np.save(save_filename+".npy", np.array(depth_pc_velo))
                depth_pc_velo = depth_pc_velo.astype(np.float32)
                depth_pc_velo.tofile(save_filename)

        color = (0, 1, 0)
        for i, obj in enumerate(objects):
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            logging.debug("box3d_pts_3d_velo - Obj. {}".format(i + 1))
            logging.debug(box3d_pts_3d_velo)

            draw_gt_boxes3d([box3d_pts_3d_velo], fig=self.fig, color=color,
                            label=str(obj.type) + '- Obj. ' + str(i + 1))

        if objects_pred is not None:
            for i, obj in enumerate(objects_pred):
                if obj.type == "DontCare":
                    continue
                if obj.score < score_threshold:
                    continue
                # Draw 3d bounding box
                _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                # color = tuple(colors[int(obj.id) % max_color])
                box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
                logging.debug("box3d_pts_3d_velo (Pred {}):".format(str(i + 1)))
                # label = str(obj.type)[:3] + ' %d' % obj.id + ': {:.1f}'.format(obj.score)
                label = str(obj.type)[:3] + ': {:.1f}'.format(obj.score)
                # draw_gt_boxes3d([box3d_pts_3d_velo], fig=self.fig, color=color, label=label)
                draw_gt_boxes3d([box3d_pts_3d_velo], fig=self.fig, label=label)
                # Draw heading arrow
                _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
                ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
                x1, y1, z1 = ori3d_pts_3d_velo[0, :]
                x2, y2, z2 = ori3d_pts_3d_velo[1, :]
                mlab.plot3d(
                    [x1, x2],
                    [y1, y2],
                    [z1, z2],
                    # color=color,
                    tube_radius=None,
                    line_width=1,
                    figure=self.fig,
                )
        # mlab.view(
        #     azimuth=180,
        #     elevation=60,
        #     focalpoint=[18, 0, 0],
        #     distance=50.0,
        #     figure=self.fig,
        # )
        module_manager = self.fig.children[0].children[0]
        module_manager.scalar_lut_manager.number_of_colors = 256
        module_manager.scalar_lut_manager.lut_mode = 'jet'
        module_manager.scalar_lut_manager.reverse_lut = True

        glyph = self.fig.children[0].children[0].children[0]
        glyph.actor.property.point_size = 3.0
        scene = self.fig.scene
        scene.camera.position = [-29.529421169679004, -0.029930304051940047, 15.629631264400999]
        scene.camera.focal_point = [18.40446156066637, -0.9930973214186383, 1.4375491165923626]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.2838516930872115, -0.002267900426086406, 0.9588655134893428]
        scene.camera.clipping_range = [0.47010602542195784, 470.10602542195784]

    def show_lidar_with_boxes(self, pc_velo, objects, calib, img_fov=False, img_width=None, img_height=None,
                              objects_pred=None, depth=None):
        """ Show all LiDAR points.
            Draw 3d box in LiDAR point cloud (in velo coord system) """

        print(("All point num: ", pc_velo.shape[0]))
        if img_fov:
            pc_velo = utils.get_lidar_in_image_fov(
                pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
            )
            print(("FOV point num: ", pc_velo.shape[0]))
        print("pc_velo", pc_velo.shape)
        draw_lidar(pc_velo, fig=self.fig)
        # pc_velo=pc_velo[:,0:3]

        color = (0, 1, 0)
        for obj in objects:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)

            draw_gt_boxes3d([box3d_pts_3d_velo], fig=self.fig, color=color)

            # Draw depth
            if depth is not None:
                # import pdb; pdb.set_trace()
                depth_pt3d = utils.depth_region_pt3d(depth, obj)
                depth_UVDepth = np.zeros_like(depth_pt3d)
                depth_UVDepth[:, 0] = depth_pt3d[:, 1]
                depth_UVDepth[:, 1] = depth_pt3d[:, 0]
                depth_UVDepth[:, 2] = depth_pt3d[:, 2]
                print("depth_pt3d:", depth_UVDepth)
                dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
                print("dep_pc_velo:", dep_pc_velo)

                draw_lidar(dep_pc_velo, fig=self.fig)

            # Draw heading arrow
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=self.fig,
            )
        if objects_pred is not None:
            color = (1, 0, 0)
            for obj in objects_pred:
                if obj.type == "DontCare":
                    continue
                # Draw 3d bounding box
                _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
                print("box3d_pts_3d_velo:")
                print(box3d_pts_3d_velo)
                draw_gt_boxes3d([box3d_pts_3d_velo], fig=self.fig, color=color)
                # Draw heading arrow
                _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
                ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
                x1, y1, z1 = ori3d_pts_3d_velo[0, :]
                x2, y2, z2 = ori3d_pts_3d_velo[1, :]
                mlab.plot3d(
                    [x1, x2],
                    [y1, y2],
                    [z1, z2],
                    color=color,
                    tube_radius=None,
                    line_width=1,
                    figure=self.fig,
                )
        mlab.show(1)

    def stat_lidar_with_boxes(self, pc_velo, objects, calib):
        """ Show all LiDAR points.
            Draw 3d box in LiDAR point cloud (in velo coord system) """

        # print(('All point num: ', pc_velo.shape[0]))

        # draw_lidar(pc_velo, fig=self.fig)
        # color=(0,1,0)
        for obj in objects:
            if obj.type == "DontCare":
                continue
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            v_l, v_w, v_h, _ = utils.get_velo_whl(box3d_pts_3d_velo, pc_velo)
            print("%.4f %.4f %.4f %s" % (v_w, v_h, v_l, obj.type))

    def show_lidar_on_image(self, pc_velo, img, calib, img_width, img_height):
        """ Project LiDAR points to image """
        img = np.copy(img)
        imgfov_pc_velo, pts_2d, fov_inds = utils.get_lidar_in_image_fov(
            pc_velo, calib, 0, 0, img_width, img_height, True
        )
        imgfov_pts_2d = pts_2d[fov_inds, :]
        imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

        import matplotlib.pyplot as plt

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        for i in range(imgfov_pts_2d.shape[0]):
            depth = imgfov_pc_rect[i, 2]
            color = cmap[int(640.0 / depth), :]
            cv2.circle(
                img,
                (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
                2,
                color=tuple(color),
                thickness=-1,
            )
        return img

    def show_lidar_topview_with_boxes(self, pc_velo, objects, calib, objects_pred=None):
        """ top_view image"""
        # print('pc_velo shape: ',pc_velo.shape)
        top_view = utils.lidar_to_top(pc_velo)
        top_image = utils.draw_top_image(top_view)
        print("top_image:", top_image.shape)

        # gt

        def bbox3d(obj):
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            return box3d_pts_3d_velo

        boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        # print("box2d BV:",boxes3d)
        lines = [obj.type for obj in objects if obj.type != "DontCare"]
        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
        )

        if objects_pred is not None:
            boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
            gt = np.array(boxes3d)
            lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
            top_image = utils.draw_box3d_on_top(
                top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
            )

        cv2.imshow("top_image", top_image)
        return top_image

    def run_pipeline(self, data_idx, seq_idx=None):
        # Load data from dataset
        objects_gt = []
        if args.split == "training":
            objects_gt = self.dataset.get_label_objects(data_idx, seq_idx)

            logging.debug("======== Objects in Ground Truth ========")
            n_obj = 0
            for obj in objects_gt:
                if obj.type != "DontCare":
                    logging.debug("=== {} object ===".format(n_obj + 1))
                    obj.print_object()
                    n_obj += 1

        objects_pred = None
        if args.show_preds:
            objects_pred = self.dataset.get_pred_objets(data_idx, seq_idx)

            if len(objects_pred) > 0:
                logging.debug("======== Predicted Objects ========")
                n_obj = 0
                for obj in objects_pred:
                    if obj.type != "DontCare":
                        logging.debug("=== {} predicted object ===".format(n_obj + 1))
                        obj.print_object()
                        n_obj += 1
        n_vec = 4
        if args.pc_label:
            n_vec = 5

        dtype = np.float32
        if args.dtype64:
            dtype = np.float64
        pc_velo = self.dataset.get_lidar(data_idx, seq_idx, dtype, n_vec)[:, 0:n_vec]
        calib = self.dataset.get_calibration(data_idx, seq_idx)
        try:
            img = self.dataset.get_image(data_idx, seq_idx)
            img_height, img_width, _ = img.shape
        except:
            logging.warning(f'Cannot load image with index {data_idx}')
            return

        logging.debug(data_idx, "image shape: ", img.shape)
        logging.debug(data_idx, "velo  shape: ", pc_velo.shape)

        if args.depth:
            depth, _ = self.dataset.get_depth(data_idx)
            print(data_idx, "depth shape: ", depth.shape)
        else:
            depth = None

        if args.stat:
            self.stat_lidar_with_boxes(pc_velo, objects_gt, calib)
            return

        # Draw 3d box in LiDAR point cloud
        if args.show_lidar_topview_with_boxes:
            # Draw lidar top view
            self.show_lidar_topview_with_boxes(pc_velo, objects_gt, calib, objects_pred)

        if args.show_image_with_boxes:
            logging.debug(data_idx, "image shape: ", img.shape)
            logging.debug(data_idx, "velo  shape: ", pc_velo.shape)

            # Draw 2d and 3d boxes on image
            img = self.show_image_with_boxes(img, objects_gt, calib, preds=objects_pred, show3d=False)

        if args.show_lidar_with_depth:
            # Draw 3d box in LiDAR point cloud
            self.show_lidar_with_depth(
                pc_velo,
                objects_gt,
                calib,
                args.img_fov,
                img_width,
                img_height,
                objects_pred,
                depth,
                constraint_box=args.const_box,
                save=args.save_depth,
                pc_label=args.pc_label,
            )
        if args.show_lidar_on_image:
            # Show LiDAR points on image.
            img = self.show_lidar_on_image(pc_velo[:, 0:3], img, calib, img_width, img_height)

        if args.save:
            seq_id_str = "{:04d}".format(int(seq_idx)) if seq_idx is not None else ""
            file_name = "{:06d}.png".format(int(data_idx))
            if 'lidar' in args.save:
                fig_path = os.path.join(str(self.lidar_output), seq_id_str, file_name)
                mlab.savefig(filename=fig_path, figure=self.fig)
            if 'camera' in args.save:
                fig_path = os.path.join(str(self.camera_output), seq_id_str, file_name)
                cv2.imwrite(filename=fig_path, img=img)

        # TODO(farzad) do more cleanup here!
        if hasattr(self, "fig"):
            mlab.clf(self.fig)

        # TODO(farzad) Followings should be adapted for parallelism
        # Creating demo video. Currently no multithreading!
        assert args.workers == 0
        if args.demo:
            if 'camera' in args.demo and 'lidar' not in args.demo:
                self.video_writer.write(img)

        if args.show_lidar_with_depth:
            mlab.show(stop=True)
            mlab.clf(figure=self.fig)
        elif args.show_image_with_boxes:
            cv2.imshow('Camera image with bounding boxes', img)
            cv2.waitKey()
        elif args.show_lidar_on_image:
            cv2.imshow("Lidar PCs on camera image", img)
            cv2.waitKey()

        if args.demo:
            if 'lidar' in args.demo and 'camera' in args.demo:
                lidar_img_paths = list(self.lidar_output.glob("lidar*"))
                lidar_img_paths.sort(key=lambda path: int(str(path.stem).split("_")[-1]))
                for lidar_img_path in tqdm.tqdm(lidar_img_paths):
                    img_id = str(lidar_img_path.stem).split("_")[-1]
                    camera_img_path = self.camera_output / f'camera_{img_id}.png'
                    lidar_img = cv2.imread(str(lidar_img_path.absolute()))
                    cam_img = cv2.imread(str(camera_img_path.absolute()))
                    img_concat = cv2.vconcat([cam_img, lidar_img])
                    self.video_writer.write(img_concat)

            elif 'lidar' in args.demo:
                lidar_img_paths = list(self.lidar_output.glob("lidar*"))
                lidar_img_paths.sort(key=lambda path: int(str(path.stem).split("_")[-1]))
                for lidar_img_path in tqdm.tqdm(lidar_img_paths):
                    lidar_img = cv2.imread(str(lidar_img_path.absolute()))
                    self.video_writer.write(lidar_img)

            self.video_writer.release()


def depth_to_lidar_format(root_dir, args):
    dataset = KITTIDataset(args)
    for data_idx in range(len(dataset)):
        # Load data from dataset
        pc_velo = dataset.get_lidar(data_idx)[:, 0:4]
        calib = dataset.get_calibration(data_idx)
        depth, _ = dataset.get_depth(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, _ = img.shape
        print(data_idx, "image shape: ", img.shape)
        print(data_idx, "velo  shape: ", pc_velo.shape)
        print(data_idx, "depth shape: ", depth.shape)
        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        # depth_height, depth_width, depth_channel = img.shape
        # print(('Image shape: ', img.shape))
        utils.save_depth(
            data_idx,
            pc_velo,
            calib,
            args.img_fov,
            img_width,
            img_height,
            depth,
            constraint_box=args.const_box,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="KIITI Object Visualization")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="data/object",
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-i",
        "--ind",
        type=int,
        default=None,
        metavar="N",
        help="only use the sample index. (default: None (all))",
    )
    parser.add_argument(
        "-p", "--show_preds", action="store_true", help="show predictions"
    )
    parser.add_argument(
        "-s",
        "--stat",
        action="store_true",
        help=" stat the w/h/l of point cloud in gt bbox",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="use training split or testing split (default: training)",
    )
    parser.add_argument(
        "-l",
        "--lidar_dir",
        type=str,
        default=None,
        metavar="N",
        help="velodyne dir (default: velodyne)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        metavar="N",
        help="image dir (default: image_2)",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default=None,
        metavar="N",
        help="label dir (default: label_2)",
    )
    parser.add_argument(
        "--calib_dir",
        type=str,
        default=None,
        metavar="N",
        help="calibration dir (default: calib)",
    )
    parser.add_argument(
        "-e",
        "--depth_dir",
        type=str,
        default=None,
        metavar="N",
        help="depth dir  (default: depth)",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        metavar="N",
        help="detection or tracking predictions dir (default: pred)",
    )
    parser.add_argument("--gen_depth", action="store_true", help="generate depth")
    parser.add_argument("--vis", action="store_true", help="show images")
    parser.add_argument("--depth", action="store_true", help="load depth")
    parser.add_argument("--img_fov", action="store_true", help="front view mapping")
    parser.add_argument("--const_box", action="store_true", help="constraint box")
    parser.add_argument(
        "--save_depth", action="store_true", help="save depth into file"
    )
    parser.add_argument(
        "--pc_label", action="store_true", help="5-verctor lidar, pc with label"
    )
    parser.add_argument(
        "--dtype64", action="store_true", help="for float64 datatype, default float64"
    )

    parser.add_argument(
        "--show_lidar_on_image", action="store_true", help="project lidar on image"
    )
    parser.add_argument(
        "--show_lidar_with_depth",
        action="store_true",
        help="--show_lidar, depth is supported",
    )
    parser.add_argument(
        "--show_image_with_boxes", action="store_true", help="show lidar"
    )
    parser.add_argument(
        "--show_lidar_topview_with_boxes",
        action="store_true",
        help="show lidar topview",
    )
    parser.add_argument(
        "--inds_file_path",
        type=str,
        default=None,
        help="only use sample indices stored in inds_file",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        nargs='+',
        help="Specify which sensor(s) to be saved. Options are \"camera\" and \"lidar\"."
    )
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        nargs='+',
        help="Specify which sensor(s) to be demonstrated. Options are \"camera\" and \"lidar\"."
    )

    parser.add_argument(
        "--tracking", action="store_true", help="If True load tracking predictions/labels."
    )

    parser.add_argument(
        "--seq_ids",
        type=int,
        default=None,
        nargs='+',
        help="only use the sequence ids. (default: None (all))"
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=0,
        help="Num. of workers for visualization. 0 means no multithreading. Demo mode only works w/o multithreading!"
    )

    args = parser.parse_args()
    if args.show_preds:
        assert os.path.exists(args.pred_dir)

    if args.vis:
        vis = Visualizer(args)
        vis.run()

    if args.gen_depth:
        depth_to_lidar_format(args.dir, args)
