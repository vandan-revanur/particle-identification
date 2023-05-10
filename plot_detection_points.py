import json
import matplotlib.pyplot as plt

with open('out/detection_points.json', 'r') as json_file:
    detection_points_data = json.load(json_file)

detection_point_coords = []
for idx, info in detection_points_data.items():
    detection_point_coords.extend(info['detection_points'])

detection_point_x_coords_2d = []
detection_point_y_coords_2d = []
detection_point_z_coords_2d = []


for dpc in detection_point_coords:
    detection_point_x_coords_2d.append(dpc[0])
    detection_point_y_coords_2d.append(dpc[1])
    detection_point_z_coords_2d.append(dpc[2])

# print(min(detection_point_z_coords_2d),max(detection_point_z_coords_2d))
plt.scatter(detection_point_x_coords_2d, detection_point_y_coords_2d)
plt.title('Detection points of particles')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.savefig('img/2d_plot_detection_points.png')
# plt.show()


# Z vs y
plt.figure()
plt.scatter(detection_point_z_coords_2d, detection_point_y_coords_2d)
plt.title('Detection points of particles: Z vs y')
plt.xlabel('Z (m)')
plt.ylabel('Y (m)')
plt.savefig('img/2d_plot_detection_points_z_vs_y.png')
# plt.show()



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(detection_point_x_coords_2d, detection_point_y_coords_2d, detection_point_z_coords_2d)
plt.title('Detection points of particles')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
# ax.set_zlim(-300, 300)
plt.savefig('img/3d_plot_detection_points.png')
# plt.show()