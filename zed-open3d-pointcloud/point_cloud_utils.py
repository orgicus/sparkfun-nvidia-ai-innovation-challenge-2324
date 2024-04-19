import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from math import radians

class PointCloudProcessor:

    def __init__(self):
        self.point_cloud_o3d = o3d.geometry.PointCloud()
        self.point_cloud_crp = o3d.geometry.PointCloud()

        """
            local 4x4 affine transformation matrix
            identity       [:3, :3]  = rotation (R) , [0:3,3] = translation (T)
            [1, 0, 0, 0,   [R, R, R, Tx,
             0, 1, 0, 0,    R, R, R, Ty,
             0, 0, 1, 0,    R, R, R, Tz,
             0, 0, 0, 1]    0, 0, 0, 1]
        """
        self.M                    = np.eye(4)
        # individual translation components (for easy serialisation)
        self.tx                   = 0
        self.ty                   = 0
        self.tz                   = 0
        # individual rotation components (for easy serialisation)
        self.rx                   = 0
        self.ry                   = 0
        self.rz                   = 0

        self.box_width            = 3.0
        self.box_height           = 0.75
        self.box_depth            = 1.2

        self.box_x                = 0.0
        self.box_y                = -0.5
        self.box_z                = -2.7

        self.box_rx               = 0
        self.box_ry               = 36
        self.box_rz               = -5
        self.crop_box             = self.make_box()

        self.res_scale            = 1
        self.depth_width          = 40
        self.depth_height         = 30
        self.res_width            = self.depth_width * self.res_scale
        self.res_height           = self.depth_height * self.res_scale
        self.num_pts              = self.res_width * self.res_height

        self.use_rgb              = False
        self.point_size           = 3.0
        self.hide_original        = True
        self.hide_cropped         = False

        self.cluster_min_points   = 10
        self.cluster_eps          = 0.45
        self.box_extent_threshold = 0.7

        self.frame_count          = 0
        self.prev_bbs             = []
        self.use_point_clustering = False

        self.vis                  = None

        self.amountIncrement      = 0.1

    def move(self, x, y, z):
        """
            applies relative offset to the original and cropped point clouds
        """
        self.tx += x
        self.ty += y
        self.tz += z
        self.M[:3,3] = (self.tx, self.ty, self.tz)

    def rotate(self, delta_rx, delta_ry, delta_rz):
        """
            applies relative rotation to the original and cropped point clouds

            expects angles in degrees
        """
        self.rx += delta_rx
        self.ry += delta_ry
        self.rz += delta_rz
        self.M[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([radians(self.rx),radians(self.ry), radians(self.rz)])

    def make_box(self):
        """
            makes an oriented bounding box to crop the point cloud by
        """
        box = o3d.geometry.TriangleMesh.create_box(self.box_width, self.box_height, self.box_depth)
        crop_box = box.get_oriented_bounding_box(False)
        crop_box.color = [0, 0, 0]
        crop_box.translate([self.box_x, self.box_y, self.box_z])
        crop_box.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([radians(self.box_rx),radians(self.box_ry), radians(self.box_rz)]))
        print("dbg", crop_box,self.box_depth, self.box_width, self.box_height)
        return crop_box

    def get_clusters(self, pcd, apply_label_colors = True, compute_aabbs = True, eps=0.45, min_points=10):
        """
            pcd - the point cloud to process

            apply_label_colors - if true, uses plt to generate a colour map for the labels

            compute_aabbs - if true, computes the axis aligned bounding box (aabb) of each point cloud cluster

            eps - db scan algorith eps ratio (sensitivity)

            min_points - the minimum number of points that can be considered a cluster
            use a small number for sparse point clouds / a larger number for dense point clouds
        """
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        
        if labels.size == 0:
            return None, None

        max_label = labels.max()

        if apply_label_colors:
            colors = plt.get_cmap("tab20")(labels / max(max_label,1))
            colors[labels < 0] = 0
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        clusters = [pcd.select_by_index(list(np.where(labels == i)[0])) for i in range(max_label + 1)]
        aabbs    = [cluster.get_axis_aligned_bounding_box() for cluster in clusters] if compute_aabbs else []

        for aabb in aabbs:
            aabb.color = [0, 0, 0]

        return clusters, aabbs

    def on_box_w_minus(self, vis):
        amount = -self.amountIncrement
        self.box_width += amount
        self.update_box(vis)

    def on_box_w_plus(self, vis):
        amount = self.amountIncrement
        self.box_width += amount
        self.update_box(vis)

    def on_box_h_minus(self, vis):
        amount = -self.amountIncrement
        self.box_height += amount
        self.update_box(vis)

    def on_box_h_plus(self, vis):
        amount = +self.amountIncrement
        self.box_height += amount
        self.update_box(vis)

    def on_box_d_minus(self, vis):
        amount = -self.amountIncrement
        self.box_depth += amount
        self.update_box(vis)

    def on_box_d_plus(self, vis):
        amount = +self.amountIncrement
        self.box_depth += amount
        self.update_box(vis)

    def on_box_x_minus(self, vis):
        amount = -self.amountIncrement
        self.box_x += amount
        self.crop_box.translate([amount, 0, 0])

    def on_box_x_plus(self, vis):
        amount = +self.amountIncrement
        self.box_x += amount
        self.crop_box.translate([amount, 0, 0])

    def on_box_y_minus(self, vis):
        amount = -self.amountIncrement
        self.box_y += amount
        self.crop_box.translate([0, amount, 0])

    def on_box_y_plus(self, vis):
        amount = +self.amountIncrement
        self.box_y += amount
        self.crop_box.translate([0, amount, 0])

    def on_box_z_minus(self, vis):
        amount = -self.amountIncrement
        self.box_z += amount
        self.crop_box.translate([0, 0, amount])

    def on_box_z_plus(self, vis):
        amount = +self.amountIncrement
        self.box_z += amount
        self.crop_box.translate([0, 0, amount])

    def on_box_rx_plus(self, vis):
        amount = 1
        self.box_rx += amount
        self.crop_box.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([radians(amount),radians(0), radians(0)]))

    def on_box_rx_minus(self, vis):
        amount = -1
        self.box_rx += amount
        self.crop_box.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([radians(amount),radians(0), radians(0)]))

    def on_box_ry_plus(self, vis):
        amount = 1
        self.box_ry += amount
        self.crop_box.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([radians(0),radians(amount), radians(0)]))

    def on_box_ry_minus(self, vis):
        amount = -1
        self.box_ry += amount
        self.crop_box.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([radians(0),radians(amount), radians(0)]))

    def on_box_rz_plus(self, vis):
        amount = 1
        self.box_rz += amount
        self.crop_box.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([radians(0),radians(0), radians(amount)]))

    def on_box_rz_minus(self, vis):
        amount = -1
        self.box_rz += amount
        self.crop_box.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([radians(0),radians(0), radians(amount)]))

    def update_box(self, vis):
        if vis != None:
            vis.remove_geometry(self.crop_box, reset_bounding_box = False)
        self.crop_box = self.make_box()
        if vis != None:
            vis.add_geometry(self.crop_box, reset_bounding_box = False)
            print(self.crop_box)

    def update_res(self):
        self.res_width    = self.depth_width * self.res_scale
        self.res_height   = self.depth_height * self.res_scale
        self.num_pts      = self.res_width * self.res_height

    def on_res_up(self, vis):
        self.res_scale += 1
        self.update_res()

    def on_res_down(self, vis):
        self.res_scale -= 1
        self.res_scale = max(self.res_scale, 1)
        self.update_res()

    def on_rgb(self, vis):
        self.use_rgb = not self.use_rgb

    def on_hide_original(self, vis):
        self.hide_original = not self.hide_original

    def on_hide_cropped(self, vis):
        self.hide_cropped = not self.hide_cropped


    def on_point_size_up(self, vis):
        self.point_size += 1
        if vis != None:
            vis.get_render_option().point_size = self.point_size

    def on_point_size_down(self, vis):
        self.point_size -= 1
        if vis != None:
            vis.get_render_option().point_size = self.point_size

    def on_cluster_min_points_minus(self, vis):
        self.cluster_min_points -= 1
        self.cluster_min_points = max(self.cluster_min_points, 1)

    def on_cluster_min_points_plus(self, vis):
        self.cluster_min_points += 1

    def on_cluster_eps_plus(self, vis):
        self.cluster_eps += 0.05

    def on_cluster_eps_minus(self, vis):
        self.cluster_eps -= 0.05
        self.cluster_eps = max(self.cluster_eps, 0.05)

    def on_box_extent_threshold_minus(self, vis):
        self.box_extent_threshold -= 0.1
        self.box_extent_threshold = max(self.box_extent_threshold, 0.1)

    def on_box_extent_threshold_plus(self, vis):
        self.box_extent_threshold += 0.1

    def get_settings(self):
        return {
            'box_width'     : self.box_width,
            'box_height'    : self.box_height,
            'box_depth'     : self.box_depth,
            'box_x'         : self.box_x,
            'box_y'         : self.box_y,
            'box_z'         : self.box_z,
            'box_rx'        : self.box_rx,
            'box_ry'        : self.box_ry,
            'box_rz'        : self.box_rz,
            'hide_original' : self.hide_original,
            "tx"            : self.tx,
            "ty"            : self.ty,
            "tz"            : self.tz,
            "rx"            : self.rx,
            "ry"            : self.ry,
            "rz"            : self.rz,
            }

    def set_settings(self, new_settings):
        self.box_width  = new_settings['box_width']
        self.box_height = new_settings['box_height']
        self.box_depth  = new_settings['box_depth']
        self.box_x      = new_settings['box_x']
        self.box_y      = new_settings['box_y']
        self.box_z      = new_settings['box_z']
        self.box_rx     = new_settings['box_rx']
        self.box_ry     = new_settings['box_ry']
        self.box_rz     = new_settings['box_rz']
        self.update_box(self.vis)
        self.tx         = new_settings['tx']
        self.ty         = new_settings['ty']
        self.tz         = new_settings['tz']
        self.rx         = new_settings['rx']
        self.ry         = new_settings['ry']
        self.rz         = new_settings['rz']
        # apply settings to transformation matrix
        self.M[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([radians(self.rx),radians(self.ry), radians(self.rz)])
        self.M[0:3,3] = (self.tx, self.ty, self.tz)


if __name__ == '__main__':
    # basic tests
    pcp = PointCloudProcessor()
    print(pcp.crop_box)
    # test W, H, D
    print("W, H, D")
    pcp.on_box_w_minus(None)
    print(pcp.crop_box)
    pcp.on_box_w_plus(None)
    print(pcp.crop_box)
    pcp.on_box_h_minus(None)
    print(pcp.crop_box)
    pcp.on_box_h_plus(None)
    print(pcp.crop_box)
    pcp.on_box_d_minus(None)
    print(pcp.crop_box)
    pcp.on_box_d_plus(None)
    print(pcp.crop_box)
    # test X, Y, Z
    print("X, Y, Z")
    pcp.on_box_x_minus(None)
    print(pcp.crop_box)
    pcp.on_box_x_plus(None)
    print(pcp.crop_box)
    pcp.on_box_y_minus(None)
    print(pcp.crop_box)
    pcp.on_box_y_plus(None)
    print(pcp.crop_box)
    pcp.on_box_z_minus(None)
    print(pcp.crop_box)
    pcp.on_box_z_plus(None)
    print(pcp.crop_box)
    # test RX, RY, RZ
    print("RX, RY, RZ")
    pcp.on_box_rx_minus(None)
    print(pcp.crop_box, np.asarray(pcp.crop_box.get_box_points()))
    pcp.on_box_rx_plus(None)
    print(pcp.crop_box, np.asarray(pcp.crop_box.get_box_points()))
    pcp.on_box_ry_minus(None)
    print(pcp.crop_box, np.asarray(pcp.crop_box.get_box_points()))
    pcp.on_box_ry_plus(None)
    print(pcp.crop_box, np.asarray(pcp.crop_box.get_box_points()))
    pcp.on_box_rz_minus(None)
    print(pcp.crop_box, np.asarray(pcp.crop_box.get_box_points()))
    pcp.on_box_rz_plus(None)
    print(pcp.crop_box, np.asarray(pcp.crop_box.get_box_points()))
    # test res
    pcp.on_res_up(None)
    print("res", pcp.res_width, pcp.res_height, pcp.num_pts)
    pcp.on_res_down(None)
    print("res", pcp.res_width, pcp.res_height, pcp.num_pts)
    # test vis vars
    print('use_rgb', pcp.use_rgb)
    pcp.on_rgb(None)
    print('use_rgb', pcp.use_rgb)
    print('hide_original', pcp.hide_original)
    pcp.on_hide_original(None)
    print('hide_original', pcp.hide_original)
    print('point_size', pcp.point_size)
    pcp.on_point_size_up(None)
    print('point_size', pcp.point_size)
    pcp.on_point_size_down(None)
    print('point_size', pcp.point_size)
    # test clustering vars
    print('cluster_min_points', pcp.cluster_min_points)
    pcp.on_cluster_min_points_minus(None)
    print('cluster_min_points', pcp.cluster_min_points)
    pcp.on_cluster_min_points_plus(None)
    print('cluster_min_points', pcp.cluster_min_points)
    print('cluster_eps', pcp.cluster_eps)
    pcp.on_cluster_eps_plus(None)
    print('cluster_eps', pcp.cluster_eps)
    pcp.on_cluster_eps_minus(None)
    print('cluster_eps', pcp.cluster_eps)
    print('box_extent_threshold', pcp.box_extent_threshold)
    pcp.on_box_extent_threshold_minus(None)
    print('box_extent_threshold', pcp.box_extent_threshold)
    pcp.on_box_extent_threshold_minus(None)
    print('box_extent_threshold', pcp.box_extent_threshold)
