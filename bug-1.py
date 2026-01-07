# Paste and run in Spyder (run "%matplotlib qt" first)
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import Point, Polygon

# -------------------------
# Parameters
# -------------------------
robot_radius = 0.5
sensor_range = 2.0

# -------------------------
# Utility helpers
# -------------------------
def move_towards(p1, p2, step_size):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    d = math.hypot(dx, dy)
    if d == 0:
        return (p2[0], p2[1])
    return (p1[0] + dx / d * step_size, p1[1] + dy / d * step_size)

def to_tuple_f(p):
    return (float(p[0]), float(p[1]))

# -------------------------
# Bug-1 Continuous with Disk Robot
# -------------------------
def simulate_bug1_stepwise(start, goal, obstacles, step_size=0.25, max_iters=50000):
    # Inflate obstacles to account for robot radius
    inflated_obstacles = [obs.buffer(robot_radius) for obs in obstacles]

    path = [to_tuple_f(start)]
    events = []
    current = Point(start)
    goal_pt = Point(goal)
    it = 0

    while current.distance(goal_pt) > step_size and it < max_iters:
        it += 1

        # Step toward goal
        next_xy = move_towards((current.x, current.y), (goal[0], goal[1]), step_size)
        next_pt = Point(next_xy)

        # Check for collision with inflated obstacles
        hit_obs = None
        for i, obs in enumerate(inflated_obstacles):
            if obs.contains(next_pt):
                hit_obs = (i, obs)
                break

        if hit_obs is None:
            current = next_pt
            path.append(to_tuple_f((current.x, current.y)))
            continue

        # ---------- Hit obstacle ----------
        obs_idx, obs = hit_obs
        print(f"> Hit obstacle {obs_idx+1} near ({current.x:.3f}, {current.y:.3f})")

        boundary = obs.exterior
        L = boundary.length

        start_s = boundary.project(current)
        start_pt = boundary.interpolate(start_s % L)
        path.append(to_tuple_f((start_pt.x, start_pt.y)))

        s = start_s
        circumnav_pts = []
        best_s = start_s % L
        best_pt = start_pt
        best_dist = best_pt.distance(goal_pt)

        while s <= start_s + L + 1e-9:
            p = boundary.interpolate(s % L)
            circumnav_pts.append(to_tuple_f((p.x, p.y)))

            d = p.distance(goal_pt)
            if d < best_dist:
                best_dist = d
                best_s = s % L
                best_pt = p

            s += step_size

        path.extend(circumnav_pts)
        print(f"> Completed circumnavigation of obstacle {obs_idx+1}. Best leave dist={best_dist:.3f} at ({best_pt.x:.3f}, {best_pt.y:.3f})")

        forward_distance = (best_s - (start_s % L)) % L
        leave_pts = []
        if forward_distance > 1e-8:
            s2 = start_s
            traveled = 0.0
            while traveled < forward_distance - 1e-9:
                seg = min(step_size, forward_distance - traveled)
                s2 += seg
                p = boundary.interpolate(s2 % L)
                leave_pts.append(to_tuple_f((p.x, p.y)))
                traveled += seg

        path.extend(leave_pts)
        current = Point(float(best_pt.x), float(best_pt.y))
        path.append(to_tuple_f((current.x, current.y)))
        events.append((len(path)-1, obs_idx))
        print(f"> Leaving obstacle {obs_idx+1} at ({current.x:.3f}, {current.y:.3f})")

    if Point(path[-1]).distance(goal_pt) > 1e-6:
        path.append(to_tuple_f(goal))

    print(f"Simulation finished: {len(path)} points, iterations={it}")
    return path, events, inflated_obstacles

# -------------------------
# Obstacles (Case 1)
# -------------------------
new_obstacle_coords = [
    (-8.052000, -6.720000),
    (4.576000, 7.933333),
    (1.408000, 8.353333),
    (-11.000000, -5.040000)
]

obstacles = [Polygon(new_obstacle_coords)]

# -------------------------
# Start & Goal
# -------------------------
start = (-15, -10)
goal = (15, 10)

# -------------------------
# Simulate
# -------------------------
path, events, inflated_obstacles = simulate_bug1_stepwise(start, goal, obstacles, step_size=0.25)

# -------------------------
# Animate in Spyder
# -------------------------
fig, ax = plt.subplots(figsize=(12,9))

# Workspace
ax.plot([-20, 20, 20, -20, -20], [15, 15, -15, -15, 15], 'k--', alpha=0.4)

# Obstacles (original + inflated)
ox, oy = obstacles[0].exterior.xy
ax.fill(ox, oy, alpha=0.6, color='lightcoral', label='Original Obstacle')

iox, ioy = inflated_obstacles[0].exterior.xy
ax.fill(iox, ioy, alpha=0.2, color='gray', hatch='//', label='Inflated (robot radius)')

# Start, goal, path
ax.scatter(start[0], start[1], s=120, c='green', edgecolors='black', label='Start')
ax.scatter(goal[0], goal[1], s=140, c='red', marker='*', edgecolors='black', label='Goal')

line, = ax.plot([], [], lw=2, color='blue', label='Path')
robot_dot, = ax.plot([], [], 'o', color='navy', markersize=10, label='Robot')

# Sensor range visualization
sensor_circle = plt.Circle(start, sensor_range, color='cyan', alpha=0.2)
ax.add_patch(sensor_circle)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
ax.set_xlim(-20, 20)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Bug1 (Disk Robot) – Case 1")

px = [float(p[0]) for p in path]
py = [float(p[1]) for p in path]

def init():
    line.set_data([], [])
    robot_dot.set_data([], [])
    sensor_circle.center = start
    return [line, robot_dot, sensor_circle]

def update(i):
    line.set_data(px[:i+1], py[:i+1])
    robot_dot.set_data([px[i]], [py[i]])
    sensor_circle.center = (px[i], py[i])
    return [line, robot_dot, sensor_circle]

ani = animation.FuncAnimation(fig, update, frames=len(path), init_func=init,
                              interval=30, blit=False, repeat=False)
#ani.save("bug1_case1.gif", writer='pillow', fps=20)


plt.show()
