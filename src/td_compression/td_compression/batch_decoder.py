import rclpy
from rclpy.node import Node
from octomap_msgs.msg import Octomap
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import torch
import yaml
import pickle
import os
import glob
from visualization_msgs.msg import Marker, MarkerArray
from ament_index_python.packages import get_package_share_directory
from td_compression.models.vae_module import LitVAE
import octomap_bridge

class BatchDecoder(Node):
    def __init__(self):
        super().__init__('batch_decoder_node')
        
        pkg_share = get_package_share_directory('td_compression')

        default_config = os.path.join(pkg_share, 'config', 'SR_128.yaml')
        default_ckpt = os.path.join(pkg_share, 'weights', 'SR_128.ckpt')
        default_in = "/home/hampek/uni/x7014e/data/keyframes/sr_B_route2/SR_128"

        self.declare_parameter('config_path', default_config)
        self.declare_parameter('model_path', default_ckpt)
        self.declare_parameter('input_dir', default_in)

        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.input_dir = self.get_parameter('input_dir').get_parameter_value().string_value


        meta_path = os.path.join(self.input_dir, "metadata.yaml")
        with open(meta_path, 'r') as f:
            meta = yaml.safe_load(f)
            self.res = meta['resolution']
            self.grid_size = meta['grid_size']
            self.z_layers = meta['z_layers']
            self.fixed_frame = meta['frame_id']
        
        self.octo_manager = octomap_bridge.OctomapManager(self.res)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # model can be either .ckpt or .pt
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if self.model_path.endswith('.ckpt'):
                self.model = LitVAE.load_from_checkpoint(self.model_path, config=config).to(self.device)
            elif self.model_path.endswith(('.pt', '.pth')):
                self.model = LitVAE(config).to(self.device)
                state_dict = torch.load(self.model_path, map_location=self.device)
                
                if 'state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['state_dict'])
                else:
                    self.model.load_state_dict(state_dict)
            else:
                raise ValueError("Model path must be a .ckpt or .pt file")

            self.model.eval()
            self.get_logger().info("VAE Model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load VAE: {e}")
            raise


        self.keyframe_files = sorted(glob.glob(os.path.join(self.input_dir, "*.npz")))
        self.current_idx = 0

        self.marker_pub = self.create_publisher(MarkerArray, '/keyframe_markers', 10)
        self.octo_pub = self.create_publisher(Octomap, '/reconstructed_octomap', 10)
        self.pc_pub = self.create_publisher(PointCloud2, '/reconstructed_points', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.bt_filepath = os.path.join(self.input_dir, "reconstructed_cave.bt")

        self.playback_rate = 0.1 # rviz cant keep up :(
        self.timer = self.create_timer(self.playback_rate, self.process_next_keyframe)

        self.map_has_data = False
        self.map_timer = self.create_timer(5.0, self.publish_current_map)
    def process_next_keyframe(self):
        if not self.keyframe_files:
            self.get_logger().warn("No keyframes found!!")
            self.timer.cancel()
            return

        if self.current_idx < len(self.keyframe_files):
            # if self.current_idx > 9:
            #     return
            filepath = self.keyframe_files[self.current_idx]
            self.decode_single(filepath)
        else:
            self.save_map()
            self.timer.cancel()

    def decode_single(self, filepath):
        data = np.load(filepath)
        latent_vector = data['latent']
        tx, ty, tz = data['position']

        current_time = self.get_clock().now().to_msg()


        t = TransformStamped()
        t.header.stamp, t.header.frame_id = current_time, self.fixed_frame 
        t.child_frame_id = 'reconstruction_sensor'
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = tz

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        self.publish_rviz_markers(tx, ty, tz, self.current_idx)

        latent_tensor = torch.from_numpy(latent_vector).float().to(self.device).view(1, -1)
        with torch.no_grad():
            outputs = self.model.decoder(latent_tensor)
            pred_binary = (torch.sigmoid(outputs).squeeze(0).cpu().numpy() > 0.5).astype(np.uint32)


        z_idx, y_idx, x_idx = np.nonzero(pred_binary)
        
        if len(z_idx) > 0:
            local_x = (x_idx - (self.grid_size / 2.0)) * self.res + (self.res / 2.0)
            local_y = (y_idx - (self.grid_size / 2.0)) * self.res + (self.res / 2.0)
            local_z = (z_idx - (self.z_layers / 2.0)) * self.res + (self.res / 2.0)
            
            local_pts = np.column_stack((local_x, local_y, local_z)).astype(np.float32) # Nx3 mtx of points
            
            # pc_header = Header(stamp=current_time, frame_id='reconstruction_sensor')
            # self.pc_pub.publish(pc2.create_cloud_xyz32(pc_header, np.ascontiguousarray(local_pts)))
            

            global_pts = local_pts + np.array([tx, ty, tz], dtype=np.float32) # tf to global is just translation
            global_pts = np.ascontiguousarray(global_pts)

            self.octo_manager.inject_points(global_pts)
            self.map_has_data = True

            self.publish_current_map()
            # map_bytes = self.octo_manager.get_serialized_map()
            # msg = Octomap(header=Header(stamp=current_time, frame_id=self.fixed_frame), 
            #             binary=True, id="OcTree", resolution=self.res)
            # msg.data = np.frombuffer(map_bytes, dtype=np.int8).tolist()
            # self.octo_pub.publish(msg)

        self.current_idx += 1
        self.get_logger().info(f"Reconstructed {self.current_idx}/{len(self.keyframe_files)}.")

    def publish_current_map(self):
        if not self.map_has_data:
            return
            
        current_time = self.get_clock().now().to_msg()
        map_bytes = self.octo_manager.get_serialized_map()
        
        msg = Octomap(header=Header(stamp=current_time, frame_id=self.fixed_frame), 
                    binary=True, id="OcTree", resolution=self.res)
        msg.data = np.frombuffer(map_bytes, dtype=np.int8).tolist()
        self.octo_pub.publish(msg)

    def publish_rviz_markers(self, tx, ty, tz, kf_id):
        now = self.get_clock().now().to_msg()
        marker_array = MarkerArray()
        
        # Center Marker
        center_marker = Marker()
        center_marker.header.frame_id = self.fixed_frame
        center_marker.header.stamp = now
        center_marker.ns = "keyframe_positions"
        center_marker.id = kf_id
        center_marker.type = Marker.SPHERE
        center_marker.action = Marker.ADD
        center_marker.pose.position.x = tx
        center_marker.pose.position.y = ty
        center_marker.pose.position.z = tz
        center_marker.pose.orientation.w = 1.0
        center_marker.scale.x = 0.5
        center_marker.scale.y = 0.5
        center_marker.scale.z = 0.5
        center_marker.color.r = 1.0
        center_marker.color.a = 1.0
        
        # Coverage Box
        box_marker = Marker()
        box_marker.header.frame_id = self.fixed_frame
        box_marker.header.stamp = now
        box_marker.ns = "keyframe_coverage_boxes"
        box_marker.id = kf_id 
        box_marker.type = Marker.CUBE
        box_marker.action = Marker.ADD
        box_marker.pose.position.x = tx
        box_marker.pose.position.y = ty
        box_marker.pose.position.z = tz
        box_marker.pose.orientation.w = 1.0
        box_marker.scale.x = self.grid_size * self.res
        box_marker.scale.y = self.grid_size * self.res
        box_marker.scale.z = self.z_layers * self.res
        box_marker.color.r = 0.2
        box_marker.color.g = 0.6
        box_marker.color.b = 1.0
        box_marker.color.a = 0.15 

        marker_array.markers.extend([center_marker, box_marker])
        self.marker_pub.publish(marker_array)


    def save_map(self):
        self.octo_manager.save_map(self.bt_filepath)
        self.get_logger().info(f"Saved global map to {self.bt_filepath}")

def main(args=None):
    rclpy.init(args=args)
    node = BatchDecoder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()