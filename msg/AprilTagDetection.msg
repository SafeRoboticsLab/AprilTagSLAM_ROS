# Tag ID
int32 id

# Tag size
float64 size

# Indicator of if this tag is static
bool static_tag

# Pose in the camera frame, obtained from homography transform. 
geometry_msgs/msg/pose pose

geometry_msgs/msg/point center      # center in (x,y) pixel coordinates
geometry_msgs/msg/point[4] corners  # corners of tag ((x1,y1),(x2,y2),...)