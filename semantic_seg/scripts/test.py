#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import torch
import sparseconvnet as scn
import torch.nn as nn
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"
print(device)

cloud_points = 0


def augment_and_scale_3d(points, scale, full_scale,
    noisy_rot=0.0,
    flip_x=0.0,
    flip_y=0.0,
    rot_z=0.0,
    transl=False):
    if noisy_rot > 0 or flip_x > 0 or flip_y > 0 or rot_z > 0:
        rot_matrix = np.eye(3, dtype=np.float32)
        if noisy_rot > 0:
            rot_matrix += np.random.randn(3, 3) * noisy_rot
        if flip_x > 0:
            rot_matrix[0][0] *= np.random.randint(0, 2) * 2 - 1
        if flip_y > 0:
            rot_matrix[1][1] *= np.random.randint(0, 2) * 2 - 1
        if rot_z > 0:
            # theta = np.random.rand() * rot_z
            theta = rot_z
            z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]], dtype=np.float32)
            rot_matrix = rot_matrix.dot(z_rot_matrix)
        points = points.dot(rot_matrix)

    # scale with inverse voxel size (e.g. 20 corresponds to 5cm)
    coords = points * scale
    # translate points to positive octant (receptive field of SCN in x, y, z coords is in interval [0, full_scale])
    coords -= coords.min(0)

    if transl:
        # random translation inside receptive field of SCN
        offset = np.clip(full_scale - coords.max(0) - 0.001, a_min=0, a_max=None) * np.random.rand(3)
        coords += offset

    return coords


class UNetSCN(nn.Module):
    def __init__(self,
                 in_channels=1,
                 m=16,  # number of unet features (multiplied in each layer)
                 block_reps=1,  # depth
                 residual_blocks=False,  # ResNet style basic blocks
                 full_scale=4096,
                 num_planes=7
                 ):
        super(UNetSCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = m
        n_planes = [(n + 1) * m for n in range(num_planes)]
        DIMENSION = 3
        num_classes = 5
        # print(n_planes)

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(DIMENSION, full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(DIMENSION, in_channels, m, 3, False)).add(
            scn.UNet(DIMENSION, block_reps, n_planes, residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(DIMENSION))
        self.linear = nn.Linear(m, num_classes)

    def forward(self, x):
        # print(x)
        x = self.sparseModel(x)
        # print(x)
        x = self.linear(x)
        return x



model = UNetSCN()
MODEL_PATH='/home/mpsc/Saeid_project/DVPG_workspace/ss/pretrained_model/e_149'
#MODEL_PATH='/home/mpsc/Desktop/e_latest'
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)


def read_pointcloud(msg):
    global cloud_points
    cloud_points = msg
    


class ObjectDetection(object):
    def __init__(self, pub1, pub2):
        self._pub1 = pub1
        self._pub2 = pub2
        # self._pub3 = pub3
        # self._pub4 = pub4
        # self._pub5 = pub5

    def Arr2poseArray(self, obj_poses):
        obj_PoseArray = PoseArray()
        for obj_pose in obj_poses:
            pose = Pose()
            pose.position.x = obj_pose[0]
            pose.position.y = obj_pose[1]
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            obj_PoseArray.poses.append(pose)
        return obj_PoseArray

    def callback_cloud(self, msg):
        # print('working')
        cloud_points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)), dtype =np.float32).reshape((-1, 4))
        # print(cloud_points.shape)

        points = cloud_points[:, :3]

        coords = augment_and_scale_3d(points, 20, 4096, noisy_rot=0.0, flip_x=0.0,
            flip_y=0.0, rot_z=0.0, transl=False)

        coords = coords.astype(np.int64)

        idxs = (coords.min(1) >= 0) * (coords.max(1) < 4096)

        coords = coords[idxs]
        feats = np.ones([len(idxs), 1], np.float32) 
        # print(coords.shape)
        # print(feats.shape)

        coords = torch.from_numpy(coords)
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(0)
        locs = torch.cat([coords, batch_idxs], 1)
        # print(locs.shape)

        feats= torch.from_numpy(feats)
        # print(feats.shape)

        # locs = torch.cat(locs, 0)
        # feats = torch.cat(feats, 0)
        # print(locs.shape)
        # print(feats.shape)

        # locs = torch.cat(locs, 0)
        # feats = torch.cat(feats, 0)
        batch = {'x': [locs, feats]}

        batch['x'][0] = batch['x'][0].to(device)
        batch['x'][1] = batch['x'][1].to(device)
        # print(batch['x'][0].shape)
        out = model(batch['x'])



        ab = np.array(torch.argmax(out, 1, keepdim=False).cpu())

        index = np.where(ab ==1)
        ped_points = cloud_points[index]

        index = np.where(ab ==2)
        car_points = cloud_points[index]







        # # pc_pedestrian = pc2.create_cloud(msg.header, fields, pedestrian_cloud)
        # # segmentation()
        # # bbox_3d, obj_points = segmentation(cloud_points)
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 16, PointField.FLOAT32, 1)]
        pc_pedestrian = pc2.create_cloud(msg.header, fields, ped_points)
        pc_car = pc2.create_cloud(msg.header, fields, car_points)
        # # if obj_points:
        # #     obj_types, obj_classified = classification(obj_points)
        # #     rospy.loginfo('obj_classified: {}'.format([obj_types[i] for i in obj_classified]))

        # #     if sum(obj_classified == 0) > 0:
        # #         pedestrian_points = [obj for indx, obj in enumerate(obj_points) if obj_classified[indx] == 0]
        # #         pedestrian_cloud = np.row_stack(pedestrian_points)
        # #         pc_pedestrian = pc2.create_cloud(msg.header, fields, pedestrian_cloud)
        # #     else:
        # #         pc_pedestrian = pc2.create_cloud(msg.header, fields, [])
        # #     if sum(obj_classified == 1) > 0:
        # #         car_points = [obj for indx, obj in enumerate(obj_points) if obj_classified[indx] == 1]
        # #         car_cloud = np.row_stack(car_points)
        # #         pc_car = pc2.create_cloud(msg.header, fields, car_cloud)
        # #         # publish the centers of the vehicle object as geometry_msgs.PoseArray
        # #         car_poses = [np.mean(bbox, axis=0) for indx, bbox in enumerate(bbox_3d) if obj_classified[indx] == 1]
        # #         car_PoseArray = self.Arr2poseArray(car_poses)
        # #         car_PoseArray.header = msg.header
        # #     else:
        # #         pc_car = pc2.create_cloud(msg.header, fields, [])
        # #         car_PoseArray = PoseArray()
        # #         car_PoseArray.header = msg.header
        # #     if sum(obj_classified == 2) > 0:
        # #         cyclist_points = [obj for indx, obj in enumerate(obj_points) if obj_classified[indx] == 2]
        # #         cyclist_cloud = np.row_stack(cyclist_points)
        # #         pc_cyclist = pc2.create_cloud(msg.header, fields, cyclist_cloud)
        # #     else:
        # #         pc_cyclist = pc2.create_cloud(msg.header, fields, [])
        # #     if sum(obj_classified == 3) > 0:
        # #         misc_points = [obj for indx, obj in enumerate(obj_points) if obj_classified[indx] == 3]
        # #         misc_cloud = np.row_stack(misc_points)
        # #         pc_misc = pc2.create_cloud(msg.header, fields, misc_cloud)
        # #     else:
        # #         pc_misc = pc2.create_cloud(msg.header, fields, [])
        # # else:
        # #     pc_pedestrian = pc2.create_cloud(msg.header, fields, [])
        # #     pc_car = pc2.create_cloud(msg.header, fields, [])
        # #     pc_cyclist = pc2.create_cloud(msg.header, fields, [])
        # #     pc_misc = pc2.create_cloud(msg.header, fields, [])
        self._pub1.publish(pc_pedestrian)
        self._pub2.publish(pc_car)
        # # self._pub3.publish(pc_cyclist)
        # # self._pub4.publish(pc_misc)
        # # self._pub5.publish(car_PoseArray) 


# def main ():
    
#     rospy.spin()

if __name__ == '__main__':
    rospy.init_node('object_detection')

    pub1 = rospy.Publisher('pedestrian_points', PointCloud2, queue_size=10)
    pub2 = rospy.Publisher('car_points', PointCloud2, queue_size=10)
    # pub3 = rospy.Publisher('cyclist_points', PointCloud2, queue_size=10)
    # pub4 = rospy.Publisher('misc_points', PointCloud2, queue_size=10)
    # pub5 = rospy.Publisher('car_posearr', PoseArray, queue_size=10)
    detect = ObjectDetection(pub1, pub2)

    rospy.Subscriber("/velodyne_points", PointCloud2, read_pointcloud)
    # global cloud_points

    while not rospy.is_shutdown():
        rospy.Rate(30.0).sleep()
        if not isinstance(cloud_points, int):
            detect.callback_cloud(cloud_points)
    # main()