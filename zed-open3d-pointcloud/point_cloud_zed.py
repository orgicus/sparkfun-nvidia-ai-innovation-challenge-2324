import open3d as o3d
import numpy as np

from point_cloud_utils import PointCloudProcessor

class ZEDPointCloudProcessor(PointCloudProcessor):

    def __init__(self, zed):
        super().__init__()
        # FIXME: get depth_width, depth_height, et.c from res
        self.zed = zed
        self.depth_width  = zed.resolution.width
        self.depth_height = zed.resolution.height
        self.update_res()
        print('cc',self.num_pts)
        self.camera_source_live = self.camera_source_recorded = ""
        self.is_running = True
        self.cropped_cloud_color = np.asarray((0,.9,0))
        # list of valid (within threshold) bounding boxes after cropped pointcloud clustering
        self.valid_bbs = []

    def update_res(self):
        super().update_res()
        self.zed.update_res(self.res_width, self.res_height)
        print(self.res_width, self.res_height)

    def get_settings(self):
        settings = super().get_settings()
        settings["camera_source_recorded"] = self.camera_source_recorded
        settings["camera_source_live"] = self.camera_source_lives
        return settings

    def update_from_thread(self):
        while self.is_running:
            # print(self)
            self.update()

    def update(self, update_zed=True):
        if(update_zed): self.zed.update()
        """
        zed's get_data() returns a numpy array of shape (H, W, 4), dtype=np.float32
        o3d expects a numpy array of shape (H*W,3), dtype=np.float64

        [...,:3] returns a view of the data without the last component (e.g. (H, H, 3))
        nan_to_num cleans up the data bit: replaces nan values (copy=False means in place)
        """
        zed_xyzrgba = self.zed.point_cloud_np
        zed_xyz = zed_xyzrgba[..., :3]
        points_xyz = np.nan_to_num(zed_xyz, copy=False, nan=-1.0).reshape(self.num_pts, 3).astype(np.float64)
        self.point_cloud_o3d.points = o3d.utility.Vector3dVector(points_xyz)
        """
        zed's 4 channel contains RGBA information (4 bytes [r,g,b,a]) encoded a single 32bit float
        1. we flatten zed's 2D 4CH numpy array to a 1D 4CH numpy array : `zed_xyzrgba.reshape(self.num_pts, 4)`
        2. we grab the last channel (RGBA at index 3) `[:, 3]`
        3. we convert nan to zeros (`np.nan_to_num` with default args), with copy=True (to keep the array C-contiguous)
        4. we use `np.frombuffer` to convert each float32 to 4 bytes (np.uint8) which we reshape from flat [r0,g0,b0,a0,...] to [[r0,g0,b0,a0],...]
        5. we grab the first 3 channels: r,g,b and ignore alpha
        6. finally we convert to o3d's point cloud color format shape=(num_pixels, 3), dtype=np.float64 by casting and dividing
        """
        if self.use_rgb:
            zed_rgba = np.nan_to_num(zed_xyzrgba.reshape(self.num_pts, 4)[:, 3], copy=True)
            rgba_bytes = np.frombuffer(zed_rgba, dtype=np.uint8).reshape(self.num_pts, 4)
            points_rgb  = rgba_bytes[..., :3].astype(np.float64) / 255.0
            self.point_cloud_o3d.colors = o3d.utility.Vector3dVector(points_rgb)
        else:
            self.point_cloud_o3d.colors = ZEDPointCloudProcessor.NO_POINTS
        # tansform point clouds
        self.point_cloud_o3d.transform(self.M)

        
        if self.hide_cropped:
            self.point_cloud_crp.points = ZEDPointCloudProcessor.NO_POINTS
        else:
            self.point_cloud_crp.points = self.point_cloud_o3d.crop(self.crop_box).points
            self.point_cloud_crp.paint_uniform_color(self.cropped_cloud_color)

        if self.hide_original:
            self.point_cloud_o3d.points = ZEDPointCloudProcessor.NO_POINTS
        
        if self.use_point_clustering:
            _, bbs = self.get_clusters(self.point_cloud_crp,apply_label_colors=not self.use_rgb, \
                                                                   eps=self.cluster_eps,\
                                                                   min_points=self.cluster_min_points)
            if bbs:
                self.valid_bbs = []
                for bb in bbs:
                    if bb.get_max_extent() >= self.box_extent_threshold:
                        self.valid_bbs.append(bb)
                
    def close(self):
        if self.zed != None:
            self.zed.close()

ZEDPointCloudProcessor.NO_POINTS = o3d.utility.Vector3dVector()

if __name__ == '__main__':
    from datetime import datetime
    import sys,cv2
    from zed_utils import ZED


    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    
    if len(sys.argv) > 1:
        svo_recording_filepath = sys.argv[1]
        # open zed recording
        zed = ZED(svo_recording_filepath)
        zed.update_res(40, 30)
        # setup point cloud processor
        pcp = ZEDPointCloudProcessor(zed)
        pcp.use_rgb = True
        pcp.hide_original = False
        pcp.use_point_clustering = True 
        print(f"pcp.crop_box: {pcp.crop_box}")
        # setup Open3D visualisation
        vis = o3d.visualization.Visualizer()
        vis.create_window("ZED PointCloud Vis", width=1280, height=720)
        vis.get_render_option().line_width = 9
        # to update point clouds we need to add the geometry once
        # this is the debounce flag to do that
        was_geometry_added = False
        # keep track of cluster bbox o3d geometry so it can be removed
        prev_bbs = []

        while True:
            dt0 = datetime.now()
    
            # update processor (also updates zed cam)
            pcp.update()

            # opencv preview 2D (optional)
            cam_src = zed.camera_source
            
            rgb = zed.image.get_data()
            depth = zed.depth_view.get_data()
            cv2.imshow(f'rgb-{cam_src}', rgb)
            cv2.imshow(f'depth-{cam_src}',depth)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            
            # Open3D preview 3D
            vis.update_geometry(pcp.point_cloud_o3d)
            vis.update_geometry(pcp.point_cloud_crp)
            # add geometry once at the start
            if not was_geometry_added:
                vis.add_geometry(pcp.point_cloud_o3d)
                vis.add_geometry(pcp.point_cloud_crp)
                vis.add_geometry(pcp.crop_box)
                was_geometry_added = True
            # remove old bbs / add new ones
            for bb in prev_bbs:
                vis.remove_geometry(bb, reset_bounding_box = False)
                prev_bbs.remove(bb)

            for bb in pcp.valid_bbs:
                prev_bbs.append(bb)
                vis.add_geometry(bb, reset_bounding_box = False)
            

            # update visualiser
            vis.poll_events()
            vis.update_renderer()

            process_time = datetime.now() - dt0
            print("\rFPS: " + str(int(1 / process_time.total_seconds())), end='')
    
        # cleanup when done
        zed.close()
        cv2.destroyAllWindows()
    else:
        print("usage: pyton point_cloud_zed.py /path/to/your_zed_recording.svo")
