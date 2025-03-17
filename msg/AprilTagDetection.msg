# Tag ID
int32 id

# Tag size
float64 size

# Indicator of if this tag is static
bool static_tag

# Pose in the camera frame, obtained from homography transform. 
geometry_msgs/msg/Pose pose

geometry_msgs/msg/Point center      # center in (x,y) pixel coordinates
geometry_msgs/msg/Point[4] corners  # corners of tag ((x1,y1),(x2,y2),...)