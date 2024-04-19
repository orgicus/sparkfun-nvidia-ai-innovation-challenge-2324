import pyzed.sl as sl

class ZED:

    def __init__(self, camera_source:str) -> None:
        self.camera_source = camera_source
        self.cam = sl.Camera()
        input_type = sl.InputType()
        if camera_source.endswith(".svo"):
            input_type.set_from_svo_file(camera_source)  #Set init parameter to run from the .svo 
            self.nb_frames = self.cam.get_svo_number_of_frames()
            print("[Info] SVO contains " ,self.nb_frames," frames")
    
        # otherwise try to use string as serial number int
        else:
            input_type.set_from_serial_number(int(camera_source))

        init = sl.InitParameters(input_t=input_type, 
                        #  svo_real_time_mode=True,
                         camera_resolution=sl.RESOLUTION.VGA,
                         depth_mode=sl.DEPTH_MODE.PERFORMANCE,
                         coordinate_units=sl.UNIT.METER,
                         coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera opened succesfully 
            print("Camera Open", status, "Exit program.")
            exit(1)

        # Set a maximum resolution, for visualisation confort 
        resolution = self.cam.get_camera_information().camera_configuration.resolution
        self.resolution = sl.Resolution(min(720,resolution.width), min(404,resolution.height))
        self.init_rgbd_images_and_pointcloud()
        self.runtime = sl.RuntimeParameters()
        self.runtime.confidence_threshold = 100
        self.runtime.texture_confidence_threshold = 100
        self.runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
        self.runtime.remove_saturated_areas = True 


    def init_rgbd_images_and_pointcloud(self):
        self.image = sl.Mat(self.resolution.width, self.resolution.height, sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
        self.depth_view = sl.Mat(self.resolution.width, self.resolution.height, sl.MAT_TYPE.U8_C4)
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat(self.resolution.width, self.resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        self.point_cloud_np = None
        
    def update_res(self, res_width, res_height):
        self.resolution.width  = res_width
        self.resolution.height = res_height
        self.init_rgbd_images_and_pointcloud()
    
    def update(self, update_left_image = True, update_point_cloud = True, update_depth = True, update_depth_view = True):
        err = self.cam.grab(self.runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            if update_left_image:
                self.cam.retrieve_image(self.image,sl.VIEW.LEFT,sl.MEM.CPU,self.resolution) #retrieve image left and right
            if update_point_cloud:
                # Retrieve colored point cloud. Point cloud is aligned on the left image.
                self.cam.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, self.resolution)
                self.point_cloud_np = self.point_cloud.get_data()
            if update_depth:
                # Retrieve depth map. rescaled for visualisation
                self.cam.retrieve_image(self.depth_view, sl.VIEW.DEPTH, sl.MEM.CPU, self.resolution)
            if update_depth_view:
                # Retrieve depth map. Depth is aligned on the left image
                self.cam.retrieve_measure(self.depth, sl.MEASURE.DEPTH, sl.MEM.CPU, self.resolution)

            return True, None
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            self.cam.set_svo_position(0)
            return True, None
        else:
            return False, err
        
    def close(self):
        self.cam.close()


if __name__ == '__main__':
    import sys, logging
    from datetime import datetime
    import cv2

    if len(sys.argv) > 1:
        svo_recording_filepath = sys.argv[1]

        logging.basicConfig(filename='zed_utils_' + svo_recording_filepath + '.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        logger = logging.getLogger()
        
        frame_count = 0

        cam = ZED(svo_recording_filepath)
        
        while True:
            
            read_ok, err = cam.update()
            if read_ok:
                dt0 = datetime.now()
                rgb = cam.image.get_data()
                cv2.imshow('rgb', rgb)
                cv2.imshow('depth',cam.depth_view.get_data())
                key = cv2.waitKey(10)
                if key == ord('q'):
                    break
                
                process_time = datetime.now() - dt0
                print(f"\rFPS: {int(1 / max(process_time.total_seconds(), 0.000001))} ", end='')

                frame_count += 1

        cam.close()

