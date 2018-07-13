# coding=utf-8
# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Visualizes the object models at the ground truth poses.

# 生成适合Brachmann论文的数据集
# depth_noseg 直接使用
# info TODO 需要生成
# obj  TODO 需要生成
# rgb_noseg
# seg  TODO 渲染生成 1



import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout, misc, renderer
from params.dataset_params import get_dataset_params

dataset = 'hinterstoisser'
#dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
# dataset = 'doumanoglou'
# dataset = 'toyotalight'

# Dataset parameters
dp = get_dataset_params(dataset)

# Select IDs of scenes, images and GT poses to be processed.
# Empty list [] means that all IDs will be used.
scene_ids = [1]
im_ids = []
gt_ids = []

# Indicates whether to render RGB image
vis_rgb = True

# Indicates whether to resolve visibility in the rendered RGB image (using
# depth renderings). If True, only the part of object surface, which is not
# occluded by any other modeled object, is visible. If False, RGB renderings
# of individual objects are blended together.
vis_rgb_resolve_visib = True

# Indicates whether to render depth image
vis_depth = False

# If to use the original model color
vis_orig_color = False

# Define new object colors (used if vis_orig_colors == False)
colors = inout.load_yaml('../data/colors.yml')

# Path masks for output images
training_dataset_path = '/media/sun/Data1/Brachmann/B_type/{}/training'
test_dataset_path = '/media/sun/Data1/Brachmann/B_type/{}/test'
# Output path masks
out_rgb_mpath = training_dataset_path + '/{:02d}/rgb_noseg/color_{:05d}.png'
out_depth_mpath = training_dataset_path + '/{:02d}/depth_noseg/depth_{:05d}.png'

out_seg_mpath = training_dataset_path + '/{:02d}/seg/seg_{:05d}.png'
out_obj_mpath = training_dataset_path + '/{:02d}/obj/obj_{:05d}.png'

out_info_path = training_dataset_path + '/{:02d}/info/info_{:05}.txt'


test_rgb_mpath = test_dataset_path + '/{:02d}/rgb_noseg/color_{:05d}.png'
test_depth_mpath = test_dataset_path + '/{:02d}/depth_noseg/depth_{:05d}.png'

test_seg_mpath = test_dataset_path + '/{:02d}/seg/seg_{:05d}.png'
test_obj_mpath = test_dataset_path + '/{:02d}/obj/obj_{:05d}.png'

test_info_path = test_dataset_path + '/{:02d}/info/info_{:05}.txt'

now_test = False

# out_obj_gt_path = dataset_path+'/{:02d}/gt.yml'
# out_views_vis_mpath = dataset_path+'/views_radius={}.ply'

# base_path = "/media/sun/Data1/Brachmann/test"
# vis_rgb_mpath = base_path + 'output/vis_gt_poses_{}/{:02d}/{:04d}.jpg'
# vis_depth_mpath = base_path + '/output/vis_gt_poses_{}/{:02d}/{:04d}_depth_diff.jpg'

# Whether to consider only the specified subset of images
use_image_subset = False

# Subset of images to be considered
# 使用部分图片集合
if use_image_subset:
    im_ids_sets = inout.load_yaml(dp['test_set_fpath'])
else:
    im_ids_sets = None

scene_ids_curr = range(1, dp['scene_count'] + 1)
if scene_ids:
    scene_ids_curr = set(scene_ids_curr).intersection(scene_ids) # 求交集
for scene_id in scene_ids_curr:
    #  创建文件夹
    misc.ensure_dir(os.path.dirname(out_rgb_mpath.format(dataset, scene_id, 0)))
    misc.ensure_dir(os.path.dirname(out_depth_mpath.format(dataset, scene_id, 0)))
    misc.ensure_dir(os.path.dirname(out_seg_mpath.format(dataset, scene_id, 0)))
    misc.ensure_dir(os.path.dirname(out_obj_mpath.format(dataset, scene_id, 0)))
    misc.ensure_dir(os.path.dirname(out_info_path.format(dataset, scene_id, 0)))

    # Load scene info and gt poses
    # 加载场景信息，和gt pose
    scene_info = inout.load_info(dp['scene_info_mpath'].format(scene_id))
    scene_gt = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))
    models_info = inout.load_yaml(dp['models_info_path'])
    # Load models of objects that appear in the current scene
    # 加载模型
    obj_ids = set([gt['obj_id'] for gts in scene_gt.values() for gt in gts])
    models = {}
    for obj_id in obj_ids:
        models[obj_id] = inout.load_ply(dp['model_mpath'].format(obj_id))

    # Considered subset of images for the current scene
    if im_ids_sets is not None:
        im_ids_curr = im_ids_sets[scene_id]
    else:
        im_ids_curr = sorted(scene_info.keys())

    if im_ids:
        im_ids_curr = set(im_ids_curr).intersection(im_ids)

    # 创建数据集
    for im_id in im_ids_curr:
        print('scene: {}, im: {}'.format(scene_id, im_id))

        # Load the images
        rgb = inout.load_im(dp['test_rgb_mpath'].format(scene_id, im_id))
        depth = inout.load_depth(dp['test_depth_mpath'].format(scene_id, im_id))
        depth = depth.astype(np.float32) # [mm]
        depth *= dp['cam']['depth_scale'] # to [mm]

        # Render the objects at the ground truth poses
        im_size = (depth.shape[1], depth.shape[0])
        # ren_rgb_info = np.zeros(rgb.shape, np.uint8)
        ren_depth = np.zeros(depth.shape, np.float32)

        black_out_mask = np.zeros(depth.shape, np.float32)
        white_mask = np.ones(depth.shape, np.float32) * 255

        img_obj = np.zeros(rgb.shape,np.float32)

        gt_ids_curr = range(len(scene_gt[im_id]))
        if gt_ids:
            gt_ids_curr = set(gt_ids_curr).intersection(gt_ids)
        for gt_id in gt_ids_curr:
            gt = scene_gt[im_id][gt_id]
            obj_id = gt['obj_id']
            if vis_orig_color:
                color = (1, 1, 1)
            else:
                # color = tuple(colors[(obj_id - 1) % len(colors)]) # 从颜色表中挑颜色,这里的颜色是渲染表面颜色
                color = (1, 1, 1) # 这里我们直接使用白色
            color_uint8 = tuple([int(255 * c) for c in color])

            model = models[gt['obj_id']]
            K = scene_info[im_id]['cam_K']
            R = gt['cam_R_m2c']
            t = gt['cam_t_m2c']

            cRb = np.array([[1.0,0,0],[0,-1.0,0],[0,0,-1.0]]).astype(np.float32)
            # TODO 为适配B数据集修改
            mRmw  = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(np.float32)
            # R = [[-0.994044,0.0510079,-0.0963063],[-0.0135081,0.81922,0.573321],[0.10814,0.571207,-0.813651]]
            # R = np.array(R).reshape(3,3).astype(np.float32)
            # R = cRb.dot(R)
            # t = [-0.105358*1000,0.117521*1000,-1.01488*1000]
            # t = np.array(t).reshape(3,1).astype(np.float32)
            # t = cRb.dot(t)
            # for i in range(len(model['pts'])):
            #     model['pts'][i] = RR.dot(model['pts'][i])
            # for i in range(len(model['normals'])):
            #     model['normals'][i] = RR.dot(model['normals'][i])

            # Rendering
            if vis_rgb:
                if vis_orig_color:
                    m_rgb = renderer.render(model, im_size, K, R, t, mode='rgb')
                else:
                    m_rgb = renderer.render(model, im_size, K, R, t, mode='rgb',
                                            surf_color=color)

            if vis_depth or (vis_rgb and vis_rgb_resolve_visib):
                m_depth = renderer.render(model, im_size, K, R, t, mode='depth')

                # Get mask of the surface parts that are closer than the
                # surfaces rendered before
                visible_mask = np.logical_or(ren_depth == 0, m_depth < ren_depth)
                mask = np.logical_and(m_depth != 0, visible_mask)
                # plt.imshow(mask)
                ren_depth[mask] = m_depth[mask].astype(ren_depth.dtype)


            #################### 计算imgObj ##################
            for x in range(im_size[0]):
                for y in range(im_size[1]):
                    depth_value = ren_depth[y,x]
                    if depth_value <= 0.00000000000001:
                        img_obj[y,x,0] = 0
                        img_obj[y,x,1] = 0
                        img_obj[y,x,2] = 0
                        continue
                    pt3d = np.dot(np.linalg.inv(K), np.array([depth_value * x, depth_value * y, depth_value]))
                    # print pt3d.shape
                    # print pt3d

                    pt3d = pt3d - t.squeeze() #
                    pt3d = np.dot(R.transpose() , pt3d)
                    # 旋转到基本坐标系后，再次旋转到B坐标系
                    pt3d = mRmw.dot(pt3d)
                    img_obj[y, x, 0] = pt3d[0]
                    img_obj[y, x, 1] = pt3d[1]
                    img_obj[y, x, 2] = pt3d[2]
            #################### 计算imgObj ##################





            black_out_mask[mask] = white_mask[mask].astype(black_out_mask.dtype)

            #################### save imgs##################
            if im_id > 300:
                now_test = True
                # print "out"
            if(not now_test):

                inout.save_im(out_rgb_mpath.format(dataset, scene_id, im_id),
                              rgb.astype(np.uint8))

                inout.save_depth(out_depth_mpath.format(dataset,obj_id, im_id), depth)

                from numpngw import write_png
                write_png(out_obj_mpath.format(dataset, scene_id, im_id), img_obj.astype(np.uint16))

                inout.save_im(out_seg_mpath.format(dataset, scene_id, im_id),
                              black_out_mask.astype(np.uint8))

                R = np.dot(cRb.transpose().dot(R),mRmw.transpose())

                R_str = [[str(num) for num in item] for item in R.tolist() ]
                R_str = [" ".join(item) for item in R_str]
                R_str = [item+'\n' for item in R_str]

                t = cRb.transpose().dot(t)

                t_str = [str(item/1000)  for item in t.squeeze().tolist()]
                t_str = " ".join(t_str)

                size_x = models_info[gt['obj_id']]["size_y"]
                size_y = models_info[gt['obj_id']]["size_z"]
                size_z = models_info[gt['obj_id']]["size_x"]

                extend = [size_x, size_y, size_z]
                extend_str = [str(item/1000) for item in extend]
                extend_str = " ".join(extend_str)
                obj_name = "{:02d}"
                # def save_info_Barchmann(path, content):
                with open(out_info_path.format(dataset, scene_id, im_id), 'w') as f:
                    f.write("image size \n")
                    f.write("640 480\n")
                    f.write(str(obj_name.format(scene_id))+"\n")
                    f.write("rotation: \n")
                    f.writelines(R_str)
                    f.write("center: \n")
                    f.write(t_str+'\n')
                    f.write("extent: \n")
                    f.write(extend_str+'\n')
            else:
                if im_id > 500:
                    print "Finish job"
                    sys.exit(0)
                # print "fuck"
                misc.ensure_dir(os.path.dirname(test_rgb_mpath.format(dataset, scene_id, 0)))
                misc.ensure_dir(os.path.dirname(test_depth_mpath.format(dataset, scene_id, 0)))
                misc.ensure_dir(os.path.dirname(test_seg_mpath.format(dataset, scene_id, 0)))
                misc.ensure_dir(os.path.dirname(test_obj_mpath.format(dataset, scene_id, 0)))
                misc.ensure_dir(os.path.dirname(test_info_path.format(dataset, scene_id, 0)))

                inout.save_im(test_rgb_mpath.format(dataset, scene_id, im_id),
                              rgb.astype(np.uint8))

                inout.save_depth(test_depth_mpath.format(dataset, obj_id, im_id), depth)

                from numpngw import write_png

                write_png(test_obj_mpath.format(dataset, scene_id, im_id), img_obj.astype(np.uint16))

                inout.save_im(test_seg_mpath.format(dataset, scene_id, im_id),
                              black_out_mask.astype(np.uint8))

                R = np.dot(cRb.transpose().dot(R), mRmw.transpose())

                R_str = [[str(num) for num in item] for item in R.tolist()]
                R_str = [" ".join(item) for item in R_str]
                R_str = [item + '\n' for item in R_str]

                t = cRb.transpose().dot(t)

                t_str = [str(item / 1000) for item in t.squeeze().tolist()]
                t_str = " ".join(t_str)

                size_x = models_info[gt['obj_id']]["size_x"]
                size_y = models_info[gt['obj_id']]["size_y"]
                size_z = models_info[gt['obj_id']]["size_z"]

                extend = [size_x, size_y, size_z]
                extend_str = [str(item / 1000) for item in extend]
                extend_str = " ".join(extend_str)
                obj_name = "{:02d}"
                # def save_info_Barchmann(path, content):
                with open(test_info_path.format(dataset, scene_id, im_id), 'w') as f:
                    f.write("image size \n")
                    f.write("640 480\n")
                    f.write(str(obj_name.format(scene_id)) + "\n")
                    f.write("rotation: \n")
                    f.writelines(R_str)
                    f.write("center: \n")
                    f.write(t_str + '\n')
                    f.write("extent: \n")
                    f.write(extend_str + '\n')






        # # Save RGB visualization
        # if vis_rgb:
        #     # vis_im_rgb = 0.5 * rgb.astype(np.float32) +\
        #     #              0.5 * ren_rgb + \
        #                  # 1.0 * ren_rgb_info
        #     # vis_im_rgb[vis_im_rgb > 255] = 255
        #
        #     inout.save_im(vis_rgb_mpath.format(dataset, scene_id, im_id),
        #                   black_out_mask.astype(np.uint8))
        #
        # # Save image of depth differences
        # if vis_depth:
        #     # Calculate the depth difference at pixels where both depth maps
        #     # are valid
        #     valid_mask = (depth > 0) * (ren_depth > 0)
        #     depth_diff = valid_mask * (depth - ren_depth.astype(np.float32))
        #
        #     f, ax = plt.subplots(1, 1)
        #     cax = ax.matshow(depth_diff)
        #     ax.axis('off')
        #     ax.set_title('measured - GT depth [mm]')
        #     f.colorbar(cax, fraction=0.03, pad=0.01)
        #     f.tight_layout(pad=0)
        #     plt.savefig(vis_depth_mpath.format(dataset, scene_id, im_id), pad=0,
        #                 bbox_inches='tight')
        #     plt.close()
