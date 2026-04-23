import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point
import getpass

class ClickToGoalNode(Node):
    def __init__(self):
        super().__init__('click_to_goal_node')
        name_space = getpass.getuser()
        
        goal_point_topic = f'/{name_space}/dsp/set_goal' # dsp uses this format
        
        self.sub = self.create_subscription(PointStamped, '/clicked_point', self.click_callback, 10)
        
        self.pub = self.create_publisher(Point, goal_point_topic, 10)
        
        self.get_logger().info(f"Listening to /clicked_point, publishing to {goal_point_topic}")

    def click_callback(self, msg: PointStamped):
        goal_msg = Point()
        goal_msg.x = msg.point.x
        goal_msg.y = msg.point.y
        goal_msg.z = msg.point.z 
        
        self.pub.publish(goal_msg)
        self.get_logger().info(f"Published: x={goal_msg.x:.2f}, y={goal_msg.y:.2f}, z={goal_msg.z:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = ClickToGoalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()