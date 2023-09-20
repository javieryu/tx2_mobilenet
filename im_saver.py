import rclpy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2
import pdb

bridge = CvBridge()

def image_callback(img_msg):
    print("Recieved an image!")
    img = bridge.imgmsg_to_cv2(img_msg, "bgr8")

    cv2.imwrite("ros_img.jpg", img)
    print("wrote frame to ros_img.jpg")

def main():
    rclpy.init()
    saver_node = rclpy.create_node("im_saver")
    
    topic_name = "/image_raw"
    saver_node.create_subscription(Image, topic_name, image_callback, 1)

    rclpy.spin_once(saver_node)

    saver_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
