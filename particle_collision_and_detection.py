import pythia8
import json
import numpy as np

'''
Pythia coordinate axes convention:
The cylinder axis is in z direction(+z outside of screen), +ve x axis to the right , +ve y axis upwards
        y
        ^
        |
        |
        |
        |
        o - ---> x
      /
     /
    /
   z
'''


def cylinder_intersection(trajectory, cylinder_radius, cylinder_height):
    # assuming coordinate system: y - vertical, x - horizotal, z - beam direction

    # equation of line in x-y plane: y = ax + b
    a = (trajectory[-1][1] - trajectory[0][1]) / (trajectory[-1][0] - trajectory[0][0])
    b = trajectory[0][1] - a * trajectory[0][0]
    # equation of cylinder in x-y: x^2 + y^2 = r
    # intersection of cylinder and line in x-y: (a^2+1)x^2 + 2abx + b^2-r^2 = 0 -> Ax^2 + Bx + C = 0
    A = a ** 2 + 1
    B = 2 * a * b
    C = b ** 2 - cylinder_radius ** 2

    delta = B ** 2 - 4 * A * C
    x = 0
    if delta < 0:
        raise ValueError("Particle trajectory does not intersect cylinder")
    elif delta == 0:
        x = -1 * B / (2 * A)
    else:
        x = (-1 * B + np.sqrt(delta)) / (2 * A)
        if (trajectory[-1][0] - trajectory[0][0] < 0 and x > 0):
            x = (-1 * B - np.sqrt(delta)) / (2 * A)
        if (trajectory[-1][0] - trajectory[0][0] > 0 and x < 0):
            x = (-1 * B - np.sqrt(delta)) / (2 * A)

    y = a * x + b

    # equation of trajectory in y-z plane: y = cz + d
    c = (trajectory[-1][1] - trajectory[0][1]) / (trajectory[-1][2] - trajectory[0][2])
    d = trajectory[0][1] - a * trajectory[0][2]

    z = (y - d) / c

    return [x, y, z]


def calculate_trajectories(pythia, nsteps, ntot_particles):
    # Extract the position and momentum of each final-state particle
    print('total number of particles in the event is: ', ntot_particles)
    print('trajectory points for each particle: ', nsteps)

    x = np.zeros((ntot_particles, nsteps))
    y = np.zeros((ntot_particles, nsteps))
    z = np.zeros((ntot_particles, nsteps))
    px = np.zeros((ntot_particles, nsteps))
    py = np.zeros((ntot_particles, nsteps))
    pz = np.zeros((ntot_particles, nsteps))
    for i in range(ntot_particles):
        particle = pythia.event[i]
        if particle.isFinal() and particle.isCharged():
            p = particle.p()
            x[i][0] = particle.xProd() * 1e-3  # Convert mm/c to m/s
            y[i][0] = particle.yProd() * 1e-3
            z[i][0] = particle.zProd() * 1e-3
            px[i][0] = p.px()
            py[i][0] = p.py()
            pz[i][0] = p.pz()
            for j in range(1, nsteps):
                # Propagate the particle using the free-streaming equation
                m = p.e() ** 2 - np.dot([p.px(), p.py(), p.pz()], [p.px(), p.py(), p.pz()])
                v = pythia8.Vec4(p.px() / p.e(), p.py() / p.e(), p.pz() / p.e(), np.sqrt(m))
                x[i][j] = x[i][j - 1] + v.pT() * np.cos(v.phi()) * 1e-9  # convert GeV/c to m/s.
                y[i][j] = y[i][j - 1] + v.pT() * np.sin(v.phi()) * np.sin(v.theta()) * 1e-9
                z[i][j] = z[i][j - 1] + v.pT() * np.cos(v.theta()) * 1e-9
                px[i][j] = p.px()
                py[i][j] = p.py()
                pz[i][j] = p.pz()

    trajectories_info = {}
    for i in range(len(x)):
        if np.any(x[i]) or np.any(y[i]) or np.any(z[i]):
            for _ in range(len(x[i])):
                trajectories_info[i] = []

    for i in range(len(x)):
        if np.any(x[i]) or np.any(y[i]) or np.any(z[i]):
            for coord_x, coord_y, coord_z in zip(x[i], y[i], z[i]):
                point = np.array([coord_x, coord_y, coord_z])
                trajectories_info[i].append(point)

    trajectories_excluding_stationary = []
    for i in trajectories_info.keys():
        trajectories_info[i] = np.array(trajectories_info[i])
        trajectories_excluding_stationary.append(np.array(trajectories_info[i]))

    trajectories_excluding_stationary = np.array(trajectories_excluding_stationary)

    print('particles that had some movement/trajectory: ', len(trajectories_info))
    print('stationary particles: ', ntot_particles - len(trajectories_info))

    return trajectories_info, trajectories_excluding_stationary


def init():
    pythia = pythia8.Pythia()
    pythia.readString("Beams:eCM = 13000.")  # center-of-mass energy
    pythia.readString("HardQCD:all = on")  # turn on hard QCD processes
    pythia.readString("Random:seed = 0")  # set random seed to 0
    # set radius and length of cylindrical detector
    pythia.readString("ParticleDecays:Rmax = 2.0")
    pythia.readString("ParticleDecays:Zmax = 5.0")

    # Initialize the event generation
    pythia.init()

    # Generate an event
    if not pythia.next():
        raise RuntimeError("Event generation failed!")

    return pythia


def calculate_detection_points(cylinder_radius, cylinder_height, trajectories_excluding_stationary):
    detection_points = []
    for trajectory in trajectories_excluding_stationary:
        intersection_point = cylinder_intersection(trajectory, cylinder_radius, cylinder_height)
        detection_points.append(intersection_point)

    # print('detection_points: ',detection_points)
    detection_points = np.array(detection_points)

    return detection_points


def get_total_number_of_particles_in_event(pythia):
    ntot_particles = pythia.event.size()  # total number of particles in the event
    return ntot_particles


def analyse_starting_points_of_trajectories(trajectories_excluding_stationary):
    trajectory_starting_points = []
    for t in (trajectories_excluding_stationary):
        trajectory_starting_points.append(t[0])

    trajectory_starting_points = np.array(trajectory_starting_points)

    only_x_axis = (trajectory_starting_points[..., 0].flatten())
    only_y_axis = (trajectory_starting_points[..., 1].flatten())
    only_z_axis = (trajectory_starting_points[..., 2].flatten())

    print(min(only_x_axis))
    print(max(only_x_axis))

    print(min(only_y_axis))
    print(max(only_y_axis))

    print(min(only_z_axis))
    print(max(only_z_axis))


if __name__ == '__main__':
    # Initialize Pythia8 with default settings
    pythia = init()

    nsteps = 100  # number of steps to simulate
    cylinder_height = 10

    ntot_particles = get_total_number_of_particles_in_event(pythia)
    trajectories_info, trajectories_excluding_stationary = calculate_trajectories(pythia, nsteps, ntot_particles)

    cylinder_radii = [((i / 10) + 0.5) for i in range(10)]
    print('cylinder_radii: ', cylinder_radii)
    # cylinder_radii = range(1,101,10)

    detection_points_of_all_layers = []
    for cylinder_radius in cylinder_radii:
        detection_points = calculate_detection_points(cylinder_radius, cylinder_height,
                                                      trajectories_excluding_stationary)
        detection_points_of_all_layers.append(detection_points)

    output_info = {}
    for idx, (rad, dps) in enumerate(zip(cylinder_radii, detection_points_of_all_layers)):
        output_info[idx] = {'detection_points': dps.tolist(), 'cylinder_radius_meters': rad}

    with open('out/detection_points.json', 'w') as fp:
        json.dump(output_info, fp)
