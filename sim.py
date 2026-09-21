from dataclasses import dataclass
import math
import time
import tkinter as tk

HALF_FIELD = 72.0
ROBOT_HALF = 9.0
SENSOR_NAMES = ('Front', 'Right', 'Back', 'Left')


@dataclass
class Pose:
    x: float
    y: float
    heading: float = 0.0


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float
    name: str


# obstacles from vex high stakes
CIRCLES = [
    Circle(-67.5, 46.5, 3, 'Red upper loader'),
    Circle(-67.5, -46.5, 3, 'Red lower loader'),
    Circle(67.5, 46.5, 3, 'Blue upper loader'),
    Circle(67.5, -46.5, 3, 'Blue lower loader'),
    Circle(-21, 47.5, 4, 'Upper left leg'),
    Circle(21, 47.5, 4, 'Upper right leg'),
    Circle(-21, -47.5, 4, 'Lower left leg'),
    Circle(21, -47.5, 4, 'Lower right leg'),
    Circle(0, 0, 5, 'Center goals'),
]


def direction(heading):
    angle = math.radians(heading)
    return math.sin(angle), math.cos(angle)


def sensor_geometry(pose, index):
    dx, dy = direction(pose.heading + index * 90)
    return (pose.x + ROBOT_HALF * dx, pose.y + ROBOT_HALF * dy), (dx, dy)


def wall_hit(origin, vector):
    hits = []
    for axis in (0, 1):
        if abs(vector[axis]) > 1e-10:
            wall = math.copysign(HALF_FIELD, vector[axis])
            distance = (wall - origin[axis]) / vector[axis]
            if distance >= -1e-8:
                hits.append((max(0.0, distance), axis, wall))
    return min(hits)


def circle_hit(origin, vector, circle):
    ox, oy = origin[0] - circle.x, origin[1] - circle.y
    projection = ox * vector[0] + oy * vector[1]
    discriminant = projection ** 2 - (ox * ox + oy * oy - circle.radius ** 2)
    if discriminant < -1e-8:
        return math.inf
    root = math.sqrt(max(0, discriminant))
    hits = [t for t in (-projection - root, -projection + root) if t >= 0]
    return min(hits, default=math.inf)


def line_hit(origin, vector, start, end):
    sx, sy = end[0] - start[0], end[1] - start[1]
    det = vector[0] * sy - vector[1] * sx
    if abs(det) < 1e-10:
        return math.inf
    ax, ay = start[0] - origin[0], start[1] - origin[1]
    t = (ax * sy - ay * sx) / det
    u = (ax * vector[1] - ay * vector[0]) / det
    return t if t > 1e-8 and 0 <= u <= 1 else math.inf


def readings(pose, circles=CIRCLES):
    result = []
    for index in range(4):
        origin, vector = sensor_geometry(pose, index)
        distance = wall_hit(origin, vector)[0]
        for circle in circles:
            distance = min(distance, circle_hit(origin, vector, circle))
        result.append(distance)
    return result


class Tracker:
    def __init__(self, initial):
        self.pose = Pose(initial.x, initial.y, initial.heading)
        self.status = [''] * 4
        self.fresh = [False, False]

    def predict(self, dx, dy, heading):
        self.pose.x += dx
        self.pose.y += dy
        self.pose.heading = heading

    def update(self, distances, circles=CIRCLES, centerline=True, strict=False):
        values = [[], []]
        for index, distance in enumerate(distances):
            origin, vector = sensor_geometry(self.pose, index)
            predicted, axis, wall = wall_hit(origin, vector)
            reason = ''
            heading = (self.pose.heading + index * 90) % 90
            if not math.isfinite(distance) or distance < 0:
                reason = 'invalid'
            elif strict and distance * 25.4 > 2000:
                reason = 'range > 2000 mm'
            elif strict and 15 < heading < 75:
                reason = 'angle > 15 deg'
            elif any(circle_hit(origin, vector, c) < predicted - 1e-7 for c in circles):
                reason = 'obstacle blocked'
            # other robots are unpredictable on the other side of the field
            elif centerline and line_hit(origin, vector, (0, -72), (0, 72)) < predicted:
                reason = 'centerline filtered'
            coordinate = wall - distance * \
                vector[axis] - ROBOT_HALF * vector[axis]
            current = (self.pose.x, self.pose.y)[axis]
            if not reason and abs(coordinate - current) > 4:
                reason = 'outlier > 4 in'
            if reason:
                self.status[index] = reason
            else:
                values[axis].append(coordinate)
                self.status[index] = 'used for ' + ('X' if axis == 0 else 'Y')
        self.fresh = [bool(v) for v in values]
        if values[0]:
            self.pose.x = sum(values[0]) / len(values[0])
        if values[1]:
            self.pose.y = sum(values[1]) / len(values[1])


def corners(pose):
    angle = math.radians(pose.heading)
    c, s = math.cos(angle), math.sin(angle)
    return [(pose.x + x * c + y * s, pose.y - x * s + y * c)
            for x, y in [(-9, -9), (9, -9), (9, 9), (-9, 9)]]


def valid_pose(pose, circles=CIRCLES):
    if any(abs(x) > HALF_FIELD or abs(y) > HALF_FIELD for x, y in corners(pose)):
        return False
    angle = math.radians(pose.heading)
    c, s = math.cos(angle), math.sin(angle)
    for circle in circles:
        dx, dy = circle.x - pose.x, circle.y - pose.y
        local_x, local_y = dx * c - dy * s, dx * s + dy * c
        distance = math.hypot(max(abs(local_x) - ROBOT_HALF, 0),
                              max(abs(local_y) - ROBOT_HALF, 0))
        if distance < circle.radius:
            return False
    return True


class Simulator:
    def __init__(self, root):
        self.root = root
        root.title('VEX | Distance sensor localization')
        root.configure(bg='#17202b')
        root.resizable(False, False)
        self.keys = set()
        self.strict = tk.BooleanVar(value=False)
        self.centerline = tk.BooleanVar(value=True)
        self.canvas = tk.Canvas(
            root, width=650, height=650, bg='#17202b', highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=12, pady=12)
        panel = tk.Frame(root, bg='#17202b', padx=16, pady=20)
        panel.grid(row=0, column=1, sticky='ns')

        def label(text, size=11, color='#dbe4ed'):
            widget = tk.Label(panel, text=text, font=('Segoe UI', size), bg='#17202b',
                              fg=color, justify='left', anchor='w')
            widget.pack(fill='x', pady=5)
            return widget
        label('DISTANCE LOCALIZATION', 16, '#61d8bc')
        self.position_label = label('', 16)
        self.truth_label = label('')
        self.source_label = label('', 10, '#61d8bc')
        label('SENSOR READINGS (inches)', 11, '#61d8bc')
        self.sensor_labels = [label('', 10) for _ in range(4)]
        for text, variable in [('Autonomous centerline filter', self.centerline),
                               ('Strict angle / range limits', self.strict)]:
            tk.Checkbutton(panel, text=text, variable=variable, bg='#17202b', fg='#dbe4ed',
                           selectcolor='#263746', activebackground='#17202b',
                           activeforeground='white', takefocus=False).pack(anchor='w', pady=4)
        tk.Button(panel, text='Reset robot (R)', command=self.reset,
                  takefocus=False).pack(anchor='w', pady=8)
        root.bind('<KeyPress>', self.key_down)
        root.bind('<KeyRelease>', lambda e: self.keys.discard(e.keysym.lower()))
        root.bind('<FocusOut>', lambda e: self.keys.clear())
        self.reset()
        self.last_time = time.perf_counter()
        self.tick()

    def key_down(self, event):
        key = event.keysym.lower()
        self.keys.add(key)
        if key == 'r':
            self.reset()
        elif key == 'escape':
            self.root.destroy()

    def reset(self):
        self.truth = Pose(-42, -24)
        self.tracker = Tracker(Pose(-42, -24))
        self.imu = 0.0
        self.keys.clear()

    def advance(self, dt):
        dx = int(bool(self.keys & {'d', 'right'})) - \
            int(bool(self.keys & {'a', 'left'}))
        dy = int(bool(self.keys & {'w', 'up'})) - \
            int(bool(self.keys & {'s', 'down'}))
        turn = int('e' in self.keys) - int('q' in self.keys)
        slow = 0.3 if self.keys & {'shift_l', 'shift_r'} else 1
        length = math.hypot(dx, dy) or 1
        dx, dy = dx / length * 30 * dt * slow, dy / length * 30 * dt * slow
        rotation = turn * 100 * dt * slow
        count = max(1, math.ceil(max(abs(dx), abs(dy), abs(rotation)) / 0.4))
        for _ in range(count):
            accepted_x = accepted_y = 0.0
            for axis, amount in [('x', dx / count), ('y', dy / count), ('heading', rotation / count)]:
                candidate = Pose(self.truth.x, self.truth.y,
                                 self.truth.heading)
                setattr(candidate, axis, getattr(candidate, axis) + amount)
                candidate.heading %= 360
                if valid_pose(candidate):
                    self.truth = candidate
                    if axis == 'x':
                        accepted_x = amount
                    elif axis == 'y':
                        accepted_y = amount
                    else:
                        self.imu = (self.imu + amount) % 360
            self.tracker.predict(accepted_x,
                                 accepted_y, self.imu)
        self.distances = readings(self.truth)
        self.tracker.update(
            self.distances, centerline=self.centerline.get(), strict=self.strict.get())

    @staticmethod
    def screen(x, y):
        return 325 + x * 4, 325 - y * 4

    def draw(self):
        canvas = self.canvas
        canvas.delete('all')
        canvas.create_rectangle(
            37, 37, 613, 613, fill='#26333f', outline='#becad5', width=3)
        for coordinate in range(-48, 49, 24):
            x, y = self.screen(coordinate, coordinate)
            canvas.create_line(x, 37, x, 613, fill='#3c4a56')
            canvas.create_line(37, y, 613, y, fill='#3c4a56')
        for coordinate in range(-72, 73, 24):
            x, y = self.screen(coordinate, coordinate)
            canvas.create_text(x, 630, text=str(coordinate), fill='#a9b6c3')
            canvas.create_text(19, y, text=str(coordinate), fill='#a9b6c3')
        if self.centerline.get():
            canvas.create_line(325, 37, 325, 613,
                               fill='#e4bc68', dash=(7, 5), width=2)
        for index, distance in enumerate(self.distances):
            origin, vector = sensor_geometry(self.truth, index)
            border_distance = wall_hit(origin, vector)[0]
            blocked = any(circle_hit(origin, vector, circle) < border_distance
                          for circle in CIRCLES)
            endpoint = (origin[0] + distance * vector[0],
                        origin[1] + distance * vector[1])
            canvas.create_line(*self.screen(*origin), *self.screen(*endpoint),
                               fill='#ff4545' if blocked else '#65dce8',
                               width=2, tags='sensor_ray')
        for circle in CIRCLES:
            x, y = self.screen(circle.x, circle.y)
            radius = circle.radius * 4
            color = '#dd6d73' if circle.x < -60 else '#6c9ae5' if circle.x > 60 else '#b5a28c'
            canvas.create_oval(x-radius, y-radius, x+radius,
                               y+radius, fill=color, outline='')
        points = [v for p in corners(self.truth) for v in self.screen(*p)]
        canvas.create_polygon(points, fill='#4e8fcf',
                              outline='#b5ddff', width=2)
        dx, dy = direction(self.truth.heading)
        canvas.create_line(*self.screen(self.truth.x, self.truth.y),
                           *self.screen(self.truth.x+dx*7, self.truth.y+dy*7),
                           fill='white', arrow=tk.LAST, width=2)
        pose = self.tracker.pose
        x, y = self.screen(pose.x, pose.y)
        canvas.create_line(x-6, y, x+6, y, fill='#61ffcb', width=2)
        canvas.create_line(x, y-6, x, y+6, fill='#61ffcb', width=2)
        self.position_label.config(
            text=f'Estimated position\nX {pose.x:7.2f} in   Y {pose.y:7.2f} in')
        error = math.hypot(pose.x-self.truth.x, pose.y-self.truth.y)
        self.truth_label.config(text=f'Heading {pose.heading:6.1f} deg  |  Error {error:.2f} in\n'
                                f'Actual: ({self.truth.x:.2f}, {self.truth.y:.2f}) in')
        self.source_label.config(text='   '.join(f'{axis}: {"sensors" if fresh else "odometry"}'
                                                 for axis, fresh in zip('XY', self.tracker.fresh)))
        for name, value, status, label in zip(SENSOR_NAMES, self.distances, self.tracker.status, self.sensor_labels):
            label.config(text=f'{name:5s}  {value:7.2f} in\n{status}')

    def tick(self):
        now = time.perf_counter()
        dt = min(now - self.last_time, 0.05)
        self.last_time = now
        self.advance(dt)
        self.draw()
        self.root.after(16, self.tick)


if __name__ == '__main__':
    root = tk.Tk()
    Simulator(root)
    root.mainloop()
