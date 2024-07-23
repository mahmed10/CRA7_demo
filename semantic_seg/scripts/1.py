#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import torch
import sparseconvnet as scn
import torch.nn as nn
import pathlib

device = "cpu"
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
            theta = rot_z
            z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]], dtype=np.float32)
            rot_matrix = rot_matrix.dot(z_rot_matrix)
        points = points.dot(rot_matrix)

    coords = points * scale
    coords -= coords.min(0)

    if transl:
        offset = np.clip(full_scale - coords.max(0) - 0.001, a_min=0, a_max=None) * np.random.rand(3)
        coords += offset

    return coords

class UNetSCN(nn.Module):
    def __init__(self,
                 in_channels=1,
                 m=16,
                 block_reps=1,
                 residual_blocks=False,
                 full_scale=4096,
                 num_planes=7):
        super(UNetSCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = m
        n_planes = [(n + 1) * m for n in range(num_planes)]
        DIMENSION = 3
        num_classes = 6  # Ensure this matches your checkpoint

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(DIMENSION, full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(DIMENSION, in_channels, m, 3, False)).add(
            scn.UNet(DIMENSION, block_reps, n_planes, residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(DIMENSION))
        self.linear = nn.Linear(m, num_classes)

    def forward(self, x):
        x = self.sparseModel(x)
        x = self.linear(x)
        return x

model = UNetSCN()
project_root = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = str(project_root) + '/pretrained_model/e_149'
print(MODEL_PATH)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()  # Set the model to evaluation mode

def read_pointcloud(msg):
    global cloud_points
    cloud_points = msg
    print("Point cloud received")

class ObjectDetection(object):
    def __init__(self, pub4):
        self._pub4 = pub4

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
        # print("Processing point cloud")
        cloud_points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)), dtype=np.float32).reshape((-1, 4))
        # print(cloud_points[:. 3])

        points = cloud_points[:, :3]
        coords = augment_and_scale_3d(points, 20, 4096, noisy_rot=0.0, flip_x=0.0, flip_y=0.0, rot_z=0.0, transl=False)
        coords = coords.astype(np.int64)
        idxs = (coords.min(1) >= 0) * (coords.max(1) < 4096)
        coords = coords[idxs]
        feats = np.ones([len(idxs), 1], np.float32)

        coords = torch.from_numpy(coords)
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(0)
        locs = torch.cat([coords, batch_idxs], 1)
        feats = torch.from_numpy(feats)

        batch = {'x': [locs, feats]}
        batch['x'][0] = batch['x'][0].to(device)
        batch['x'][1] = batch['x'][1].to(device)

        with torch.no_grad():
            out = model(batch['x'])
        ab = torch.argmax(out, 1, keepdim=False).cpu().numpy()
        # print(np.unique(ab), ab.shape)

        index_person = np.where(ab == 1)
        index_car = np.where(ab == 2)
        index_bridge = np.where(ab == 3)
        index_cloud = np.where((ab == 0) | (ab > 3))

        ped_points = cloud_points[index_person]
        ped_points[:,3] = 85.0
        car_points = cloud_points[index_car]
        car_points[:,3] = 170.0
        bridge_points = cloud_points[index_bridge]
        bridge_points[:,3] = 255.0
        cloud_points = cloud_points[index_cloud]
        cloud_points[:,3] = 0.0
        cloud_points = np.vstack((cloud_points, ped_points, car_points, bridge_points))
        # print(cloud_points[:,3].dtype)
        # print(cloud_points.dtype)
        # print(np.unique(cloud_points[:,3]))
        # print(msg.header)

        # ab[ab>3] = 0
        # print(np.unique(ab), ab.dtype)
        # cloud_points[:,3] = ab*85
        # print(cloud_points[:,3].dtype)
        # print(cloud_points.shape)
        # print(np.unique(cloud_points[:,3]))


        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]
        pc_pedestrian = pc2.create_cloud(msg.header, fields, ped_points)
        pc_car = pc2.create_cloud(msg.header, fields, car_points)
        pc_bridge = pc2.create_cloud(msg.header, fields, bridge_points)
        pc_cloud = pc2.create_cloud(msg.header, fields, cloud_points)
        # print(pc_cloud)
        # print(pc_cloud)
        # print(fields)
        # print(np.max(cloud_points[:,3]))
        # print(np.min(cloud_points[:,3]))
        # print(cloud_points[:,3])
        self._pub4.publish(pc_cloud)


        
        # print(cloud_points.shape, ab.shape)
        # print("Published point clouds")

if __name__ == '__main__':
    rospy.init_node('object_detection')
    pub4 = rospy.Publisher('velodyne_points/semantic', PointCloud2, queue_size=10)

    detect = ObjectDetection(pub4)

    rospy.Subscriber("/velodyne_points", PointCloud2, read_pointcloud)

    while not rospy.is_shutdown():
        rospy.Rate(30.0).sleep()
        if isinstance(cloud_points, PointCloud2):
            detect.callback_cloud(cloud_points)


# #!/usr/bin/env python3

# import rospy
# from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2
# from geometry_msgs.msg import PoseArray, Pose
# import numpy as np
# import torch
# import sparseconvnet as scn
# import torch.nn as nn

# device = "cpu"
# print(device)

# cloud_points = 0

# def augment_and_scale_3d(points, scale, full_scale,
#     noisy_rot=0.0,
#     flip_x=0.0,
#     flip_y=0.0,
#     rot_z=0.0,
#     transl=False):
#     if noisy_rot > 0 or flip_x > 0 or flip_y > 0 or rot_z > 0:
#         rot_matrix = np.eye(3, dtype=np.float32)
#         if noisy_rot > 0:
#             rot_matrix += np.random.randn(3, 3) * noisy_rot
#         if flip_x > 0:
#             rot_matrix[0][0] *= np.random.randint(0, 2) * 2 - 1
#         if flip_y > 0:
#             rot_matrix[1][1] *= np.random.randint(0, 2) * 2 - 1
#         if rot_z > 0:
#             theta = rot_z
#             z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
#                 [np.sin(theta), np.cos(theta), 0],
#                 [0, 0, 1]], dtype=np.float32)
#             rot_matrix = rot_matrix.dot(z_rot_matrix)
#         points = points.dot(rot_matrix)

#     coords = points * scale
#     coords -= coords.min(0)

#     if transl:
#         offset = np.clip(full_scale - coords.max(0) - 0.001, a_min=0, a_max=None) * np.random.rand(3)
#         coords += offset

#     return coords

# class UNetSCN(nn.Module):
#     def __init__(self,
#                  in_channels=1,
#                  m=16,
#                  block_reps=1,
#                  residual_blocks=False,
#                  full_scale=4096,
#                  num_planes=7):
#         super(UNetSCN, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = m
#         n_planes = [(n + 1) * m for n in range(num_planes)]
#         DIMENSION = 3
#         num_classes = 5  # Ensure this matches your checkpoint

#         self.sparseModel = scn.Sequential().add(
#             scn.InputLayer(DIMENSION, full_scale, mode=4)).add(
#             scn.SubmanifoldConvolution(DIMENSION, in_channels, m, 3, False)).add(
#             scn.UNet(DIMENSION, block_reps, n_planes, residual_blocks)).add(
#             scn.BatchNormReLU(m)).add(
#             scn.OutputLayer(DIMENSION))
#         self.linear = nn.Linear(m, num_classes)

#     def forward(self, x):
#         x = self.sparseModel(x)
#         x = self.linear(x)
#         return x

# model = UNetSCN()
# MODEL_PATH = '/home/mpsc/Saeid_project/DVPG_workspace/ss/pretrained_model/e_149'
# checkpoint = torch.load(MODEL_PATH, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model = model.to(device)
# model.eval()  # Set the model to evaluation mode

# def read_pointcloud(msg):
#     global cloud_points
#     cloud_points = msg
#     print("Point cloud received")

# class ObjectDetection(object):
#     def __init__(self, pub1, pub2):
#         self._pub1 = pub1
#         self._pub2 = pub2

#     def Arr2poseArray(self, obj_poses):
#         obj_PoseArray = PoseArray()
#         for obj_pose in obj_poses:
#             pose = Pose()
#             pose.position.x = obj_pose[0]
#             pose.position.y = obj_pose[1]
#             pose.position.z = 0.0
#             pose.orientation.x = 0.0
#             pose.orientation.y = 0.0
#             pose.orientation.z = 0.0
#             pose.orientation.w = 1.0
#             obj_PoseArray.poses.append(pose)
#         return obj_PoseArray

#     def callback_cloud(self, msg):
#         print("Processing point cloud")
#         cloud_points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)), dtype=np.float32).reshape((-1, 4))

#         points = cloud_points[:, :3]
#         coords = augment_and_scale_3d(points, 20, 4096, noisy_rot=0.0, flip_x=0.0, flip_y=0.0, rot_z=0.0, transl=False)
#         coords = coords.astype(np.int64)
#         idxs = (coords.min(1) >= 0) * (coords.max(1) < 4096)
#         coords = coords[idxs]
#         feats = np.ones([len(idxs), 1], np.float32)

#         coords = torch.from_numpy(coords)
#         batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(0)
#         locs = torch.cat([coords, batch_idxs], 1)
#         feats = torch.from_numpy(feats)

#         batch = {'x': [locs, feats]}
#         batch['x'][0] = batch['x'][0].to(device)
#         batch['x'][1] = batch['x'][1].to(device)

#         with torch.no_grad():
#             out = model(batch['x'])
#         ab = torch.argmax(out, 1, keepdim=False).cpu().numpy()

#         index_person = np.where(ab == 1)
#         index_car = np.where(ab == 2)

#         ped_points = cloud_points[index_person]
#         car_points = cloud_points[index_car]

#         fields = [PointField('x', 0, PointField.FLOAT32, 1),
#                   PointField('y', 4, PointField.FLOAT32, 1),
#                   PointField('z', 8, PointField.FLOAT32, 1),
#                   PointField('intensity', 16, PointField.FLOAT32, 1)]
#         pc_pedestrian = pc2.create_cloud(msg.header, fields, ped_points)
#         pc_car = pc2.create_cloud(msg.header, fields, car_points)

#         self._pub1.publish(pc_pedestrian)
#         self._pub2.publish(pc_car)
#         print("Published point clouds")

# if __name__ == '__main__':
#     rospy.init_node('object_detection')

#     pub1 = rospy.Publisher('pedestrian_points', PointCloud2, queue_size=10)
#     pub2 = rospy.Publisher('car_points', PointCloud2, queue_size=10)
#     detect = ObjectDetection(pub1, pub2)

#     rospy.Subscriber("/velodyne_points", PointCloud2, read_pointcloud)

#     while not rospy.is_shutdown():
#         rospy.Rate(30.0).sleep()
#         if isinstance(cloud_points, PointCloud2):
#             detect.callback_cloud(cloud_points)
