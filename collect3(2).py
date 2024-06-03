import numpy as np
import quaternion
from itertools import product
class Collision:
    collision_box_shapes = np.array([  # 将机械臂等效为6个碰撞检测盒子
        [0.14553, 0.14553, 0.1313],
        [0.13644, 0.12, 0.5448],
        [0.1077, 0.116, 0.4895],
        [0.0924, 0.0785, 0.0983],
        [0.1062, 0.0785, 0.0924],
        [0.033, 0.07356, 0.08128]
    ])
    dh_params=np.array([
        [0,0.0746,np.pi/2,0],         #a,d,alpha,theta
        [-0.4251,0,0,0],
        [-0.3921,0,0,0],
        [0,0.11,np.pi/2,0],
        [0,0.0947,-np.pi/2,0],
        [0,0.075,0,0]
    ])
    num_dof=6
    _collision_box_links = [0,1,2,3,4,5]
    # _collision_box_poses_raw = np.array([                # 前三个数表示机械臂当前的位置坐标，后四位数表示机械臂的旋转矩阵
    #     [0.00428, 0, -0.00715, 0.707, 0, 0.707, 0],
    #     [-0.21255, 0, 0.13452, 0.707, 0, 0.707, 0],
    #     [0, -0.01245, -0.2055, 1, 0, 0, 0],
    #     [0, 0.0069, -0.0075, 1, 0, 0, 0],
    #     [0, 0.0036, -0.0069, 1, 0, 0, 0],
    #     [0, 0.00895, 0.00386, 1, 0, 0, 0],
    # ])  # 定义碰撞盒子的姿态
    _collision_box_poses_raw = np.array([                   #
        [-0.00718, 0, 0.07038, 1, 0, 0, 0],
        [0.21255, 0, 0.13452, 0.707, 0, 0.707, 0],
        [0.18665, 0, 0.01245, 0.707, 0, 0.707, 0],
        [-0.0075, 0, 0.1031, 0.707, 0, 0.707, 0],
        [0.0036, 0, 0.08785, 1, 0, 0, 0],
        [0.00386, 0, 0.066, 0.707, 0, 0.707, 0],
    ])  # 定义碰撞盒子的姿态
    def __init__(self):
        self._collision_boxes_data=np.zeros((len(self.collision_box_shapes),10))    #数据维度7*10   这十个维度分别储存了什么信息？
        self._collision_boxes_data[:, -3:] = self.collision_box_shapes              #将后面的三列进行填充
        self._collision_box_poses = []
        for pose in self._collision_box_poses_raw:
            T = np.eye(4)                #构建一个4*4的单位矩阵
            T[:3, 3] = pose[:3]          #获取机械臂的旋转信息
            T[:3, :3] = quaternion.as_rotation_matrix(quaternion.quaternion(*pose[3:]))     #将四元数变化为旋转矩阵
            self._collision_box_poses.append(T)
        print("=======================")
        print('self._collision_box_poses',self._collision_box_poses)
        self._vertex_offset_signs=np.array(list(product([1,-1],[1,-1],[1,-1])))             #数据维度8*3
        self._collision_box_hdiags = []
        self._collision_box_vertices_offset = []
        for sizes in self.collision_box_shapes:
            hsize=sizes/2
            self._collision_box_vertices_offset.append(self._vertex_offset_signs * hsize)         #计算碰撞检测盒子的每个顶点的位置
            self._collision_box_hdiags.append(np.linalg.norm(sizes/2))                           #计算碰撞检测盒子到每个顶点的距离
        self._collision_box_vertices_offset = np.array(self._collision_box_vertices_offset)    #数据维度7*8*3
        self._collision_box_hdiags = np.array(self._collision_box_hdiags)                      #数据维度1*7
        self._collision_proj_axes = np.zeros((3, 15))
        self._box_vertices_offset = np.ones([8, 3])
        self._box_transform = np.eye(4)                                                        # 用于创建一个4*4的单位矩阵
    def forward_kinematics(self,joints):
        forward_kinematics=np.zeros((len(self.dh_params),4,4))                                  #6*4*4
        previous_transformation=np.eye(4)
        for i in range(len(self.dh_params)):
            a,d,alpha,theta=self.dh_params[i]
            if i<self.num_dof:
                theta=theta+joints[i]
            transform=[[np.cos(theta),-np.sin(theta)*np.cos(alpha),np.sin(theta)*np.sin(alpha),a*np.cos(theta)],
                       [np.sin(theta),np.cos(theta)*np.cos(alpha),-np.cos(theta)*np.sin(alpha),a*np.sin(theta)],
                       [0,np.sin(alpha),np.cos(alpha),d],
                       [0,0,0,1]]
            forward_kinematics[i]=previous_transformation.dot(transform)
            previous_transformation=forward_kinematics[i]
        # print("==================")
        # print('forward_kinematices',forward_kinematics)
        return forward_kinematics
    #通过正运动学返回一个各个关节的一个旋转矩阵，然后用各个关节的当前位姿乘以旋转矩阵得到机械臂的各个关节在该角度下的一个姿态
    def get_collision_boxes_poses(self, joints):      #检查碰撞检测盒子的姿态是否与正运动学的一致
        fk = self.forward_kinematics(joints)          #fk的索引是0-5
        box_poses_world = []
        for i, link in enumerate(self._collision_box_links):
            link_transform = fk[link]         #将旋转矩阵赋值给相应的机械臂的关节
            print('self._collision_box_poses\n',self._collision_box_poses[i],'\n','link_transform',link_transform)
            # box_pose_world=np.matmul(self._collision_box_poses[i],link_transform)
            box_pose_world=np.matmul(link_transform,self._collision_box_poses[i])
            # box_pose_world=np.matmul(link_transform,self._collision_box_poses[i])
            # box_pose_world=link_transform.dot(self._collision_box_poses[i])
            box_poses_world.append(box_pose_world)          #得到机械臂在此关节角度下的碰撞检测姿态
        print('box_poses_world\n',box_poses_world)
        return box_poses_world
    def check_collision_box(self,joints,box):
        box_pos,box_rpy,box_hsizes = box[:3],box[3:6],box[6:]/2                                 #将盒子的姿态进行切片，将位置、欧拉角以及边长的一办获取
        box_q = quaternion.from_euler_angles(box_rpy)                                           #将欧拉角转化为4元数，便于解决万向节问题
        box_axes = quaternion.as_rotation_matrix(box_q)                                         #将四元数转化为旋转矩阵，便于解决,该结果为一个单位阵
        self._box_vertices_offset[:,:] = self._vertex_offset_signs * box_hsizes                 #障碍物的边长的一半乘以各个顶点的坐标,利用广播机制来进行矩阵乘法
        box_vertices=(box_axes.dot(self._box_vertices_offset.T)+np.expand_dims(box_pos,1)).T    #障碍物的包围盒的形状
        box_hdiag=np.linalg.norm(box_hsizes)                         #计算box_hsizes的长度
        mindistance=box_hdiag+self._collision_box_hdiags
        ur5_box_poses = self.get_collision_boxes_poses(joints)          #UR5的碰撞盒子的姿态
        for i,ur_box_pose in enumerate(ur5_box_poses):
            ubox_pos=ur_box_pose[:3,3]
            ubox_axes=ur_box_pose[:3,:3]
            if np.linalg.norm(ubox_pos-box_pos)>mindistance[i]:
                continue
            fbox_vertex_offsets = self._collision_box_vertices_offset[i]
            fbox_vertices = fbox_vertex_offsets.dot(ubox_axes.T) + ubox_pos
            cross_product_pairs = np.array(list(product(box_axes.T, ubox_axes.T)))
            cross_axes = np.cross(cross_product_pairs[:, 0], cross_product_pairs[:, 1]).T
            self._collision_proj_axes[:, :3] = box_axes
            self._collision_proj_axes[:, 3:6] = ubox_axes
            self._collision_proj_axes[:, 6:] = cross_axes
            box_projs = box_vertices.dot(self._collision_proj_axes)
            fbox_projs = fbox_vertices.dot(self._collision_proj_axes)
            min_box_projs, max_box_projs = box_projs.min(axis=0), box_projs.max(axis=0)
            min_fbox_projs, max_fbox_projs = fbox_projs.min(axis=0), fbox_projs.max(axis=0)
            if np.all([min_box_projs <= max_fbox_projs, max_box_projs >= min_fbox_projs]):
                return True
        return False
boxes = np.array([
    [0, 0, 0.05, 0, 0, 0, 0.1, 0.1, 0.1]
    ])
joints=[0, -np.pi/2, 0, -np.pi/2, 0,0]
ur5_arm=Collision()
for box in boxes:
    collision_result=ur5_arm.check_collision_box(joints,box)
    if collision_result==1:
        print("发生碰撞")
    elif collision_result==0:
        print("未发生碰撞")
