import rclpy
from rclpy.node import Node
from octomap_msgs.msg import Octomap
from visualization_msgs.msg import Marker, MarkerArray 
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
import numpy as np
import torch
import yaml
import pickle
import os
import subprocess
import math
import time 

from ament_index_python.packages import get_package_share_directory
from td_compression.models.vae_module import LitVAE
import octomap_bridge

class OnlineEncoder(Node):
    def __init__(self):
        super().__init__('online_encoder')

        self.octomap_topic = '/octomap_binary'
        self.fixed_frame = 'world'#'odom'
        self.ego_frame = 'chinook/base'#'cave_drone/base_link'
        
        pkg_share = get_package_share_directory('td_compression')
    
        default_config = os.path.join(pkg_share, 'config', 'r025_256_512x4.yaml')
        default_ckpt = os.path.join(pkg_share, 'weights', 'r025_256_512x4.ckpt')
        default_out = "/home/hampek/uni/x7014e/data/keyframes/sr_b_route2"

        self.declare_parameter('config_path', default_config)
        self.declare_parameter('model_path', default_ckpt)
        self.declare_parameter('output_dir', default_out)

        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value

        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics_filepath = os.path.join(self.output_dir, "time_metrics.csv")
        with open(self.metrics_filepath, 'w') as f:
            f.write("ext_time_ms,inf_time_ms,save_time_ms,total_time_ms\n")

        # metadata
        self.grid_size = 64
        self.z_layers = 32
        self.resolution = 0.25

        # kf selection timer
        self.check_period = 0.1 

        # thresholds        

        # determines amount of overlap in a long corridor, 
        # grid_size * resolution - hard_threshold_xy = overlap
        self.hard_threshold_xy = 14.0 
        self.hard_threshold_z = 6.0 # only ever triggers at very high elevation, only seen in sim
        self.retreat_tolerance = 2.0
        self.min_explore_dist = 5.0    
        self.revisit_tolerance_xy = 7.5
        self.revisit_tolerance_z = 3.0  
        
        self.anchor_history = []       
        self.anchor_x = self.anchor_y = self.anchor_z = None
        self.anchor_rot = self.anchor_id = None       
        self.max_dist = 0.0            
        self.ghost_pose = None         
        self.recent_poses = []
        self.latest_map_bytes = None
        self.keyframe_count = 0


        meta_path = os.path.join(self.output_dir, "metadata.yaml")
        with open(meta_path, 'w') as f:
            yaml.dump({
                "resolution": self.resolution,
                "grid_size": self.grid_size,
                "z_layers": self.z_layers,
                "frame_id": self.fixed_frame
            }, f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.sub = self.create_subscription(Octomap, self.octomap_topic, self.map_callback, map_qos)        
        self.marker_pub = self.create_publisher(MarkerArray, '/keyframe_markers', 10)
        self.timer = self.create_timer(self.check_period, self.check_pose_and_trigger)
        
        self.bt_filepath = os.path.join(self.output_dir, "ground_truth_cave.bt")

    def map_callback(self, msg):
        self.latest_map_bytes = np.array(msg.data, dtype=np.int8).tobytes()

    def check_pose_and_trigger(self):
        if self.latest_map_bytes is None:
            return

        try:
            tf = self.tf_buffer.lookup_transform(self.fixed_frame, self.ego_frame, rclpy.time.Time())
            tx, ty, tz = tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z
            rot = tf.transform.rotation
        except Exception:
            return

        # to not break with speed, query last few poses for a better kf center
        self.recent_poses.append((tx, ty, tz, rot))
        if len(self.recent_poses) > 500: # "few" xd
            self.recent_poses.pop(0)

        # first anchor
        if self.anchor_x is None:
            tx = round(tx / self.resolution) * self.resolution
            ty = round(ty / self.resolution) * self.resolution
            tz = round(tz / self.resolution) * self.resolution
            self.set_new_anchor(tx, ty, tz, rot, self.keyframe_count, is_new=True)
            return

        dx, dy, dz = tx - self.anchor_x, ty - self.anchor_y, tz - self.anchor_z
        current_xy_dist = math.hypot(dx, dy)
        current_z_dist = abs(dz)
        current_3d_dist = math.sqrt(dx**2 + dy**2 + dz**2)

        if current_3d_dist > self.max_dist:
            self.max_dist = current_3d_dist
            self.ghost_pose = (tx, ty, tz, rot)

        
        trigger = False
        target_pose = None

        if current_xy_dist >= self.hard_threshold_xy or current_z_dist >= self.hard_threshold_z:
            trigger = True
            for pose in self.recent_poses:
                # hard threshold check, cylinder area, xy matters alot more than z
                if math.hypot(pose[0]-self.anchor_x, pose[1]-self.anchor_y) >= self.hard_threshold_xy or abs(pose[2]-self.anchor_z) >= self.hard_threshold_z:
                    target_pose = pose
                    break

        # trigger at dead end, but not if robot only moves very little
        elif current_3d_dist < (self.max_dist - self.retreat_tolerance) and self.max_dist > self.min_explore_dist:
            trigger = True
            target_pose = self.ghost_pose

        if trigger:
            if target_pose is None: 
                target_pose = (tx, ty, tz, rot)
            self.process_keyframe(self.anchor_x, self.anchor_y, self.anchor_z, self.anchor_rot, self.anchor_id, reason = "Updated (leaving)")
            
            gtx, gty, gtz, grot = target_pose

            # snap to resolution s.t center is at intersection of cells
            gtx = round(gtx / self.resolution) * self.resolution
            gty = round(gty / self.resolution) * self.resolution
            gtz = round(gtz / self.resolution) * self.resolution


            # check if close enough to replace existing kf instead of making new one
            best_old = None #the best match for existing kf
            for old in self.anchor_history:
                if old[0] == self.anchor_id:
                    continue
                
                if math.hypot(gtx-old[1], gty-old[2]) < self.revisit_tolerance_xy and abs(gtz-old[3]) < self.revisit_tolerance_z:
                    best_old = old
                    break

            if best_old: 
                self.set_new_anchor(best_old[1], best_old[2], best_old[3], best_old[4], best_old[0], is_new=False)
            else:
                self.set_new_anchor(gtx, gty, gtz, grot, self.keyframe_count, is_new=True)

    def set_new_anchor(self, x, y, z, rot, kf_id, is_new):
        reason = "Created new" if is_new else "Updated (revisited)"

        self.anchor_x, self.anchor_y, self.anchor_z, self.anchor_rot, self.anchor_id = x, y, z, rot, kf_id
        self.process_keyframe(x, y, z, rot, kf_id, reason = reason)
        self.publish_rviz_markers(x, y, z, kf_id)
        if is_new:
            self.anchor_history.append((kf_id, x, y, z, rot))
            self.keyframe_count += 1
        self.max_dist = 0.0
        self.recent_poses.clear()
        

    def process_keyframe(self, tx, ty, tz, rot, kf_id, reason):
        total_start = time.perf_counter()


        t0 = time.perf_counter()
        local_grid = octomap_bridge.extract_local_grid(
            self.latest_map_bytes, tx, ty, tz, 
            0.0, 0.0, 0.0, 1.0,
            self.resolution, self.grid_size, self.z_layers
        )
        ext_time = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        input_tensor = torch.from_numpy(local_grid).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, mu, _ = self.model(input_tensor)
            latent_vector = mu.squeeze(0).cpu().numpy()

        inf_time = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        filename = os.path.join(self.output_dir, f"keyframe_{kf_id:04d}")
        np.savez(
            filename,
            latent=latent_vector,
            position=np.array([tx, ty, tz]),
            orientation=np.array([rot.x, rot.y, rot.z, rot.w]),
            frame_id=np.array([self.fixed_frame])
        )
        save_time = (time.perf_counter() - t2) * 1000.0
        total_time = (time.perf_counter() - total_start) * 1000.0

        self.get_logger().info(f"{reason} keyframe {kf_id}.")

        self.get_logger().info(f"ext:{ext_time:>5.2f}, inf:{inf_time:>5.2f}, save:{save_time:>5.2f}, tot:{total_time:>5.2f}")

        with open(self.metrics_filepath, 'a') as f:
            f.write(f"{ext_time:.2f},{inf_time:.2f},{save_time:.2f},{total_time:.2f}\n")


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
        box_marker.scale.x = self.grid_size * self.resolution
        box_marker.scale.y = self.grid_size * self.resolution
        box_marker.scale.z = self.z_layers * self.resolution
        box_marker.color.r = 0.2
        box_marker.color.g = 0.6
        box_marker.color.b = 1.0
        box_marker.color.a = 0.30

        marker_array.markers.extend([center_marker, box_marker])
        self.marker_pub.publish(marker_array)

    def save_map(self):
        if self.latest_map_bytes is not None:
            try:
                octomap_bridge.save_map(self.latest_map_bytes, self.resolution, self.bt_filepath)
                self.get_logger().info(f"Saved ground truth map to {self.bt_filepath}")
            except Exception as e:
                self.get_logger().error(f"Failed to save map via bridge: {e}")
def main(args=None):
    rclpy.init(args=args)
    node = OnlineEncoder()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.save_map()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()