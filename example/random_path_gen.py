import numpy as np

def generate_trapezoid_vertices(FOV, near_distance, far_distance, offset_x=0, offset_y=0):
    """
    根據 FOV 和剪裁距離計算視錐梯形的頂點座標，並可設置偏移量。
    """
    FOV_rad = np.radians(FOV / 2)
    L_near = 2 * near_distance * np.tan(FOV_rad)
    L_far = 2 * far_distance * np.tan(FOV_rad)

    P1 = np.array([-L_near / 2 + offset_x, near_distance + offset_y])
    P2 = np.array([L_near / 2 + offset_x, near_distance + offset_y])
    P3 = np.array([-L_far / 2 + offset_x, far_distance + offset_y])
    P4 = np.array([L_far / 2 + offset_x, far_distance + offset_y])

    return P1, P2, P3, P4

def generate_random_point_on_edge(P1, P2):

    t = np.random.uniform(0, 1)  # 隨機比例
    return (1 - t) * P1 + t * P2

def generate_random_point_on_edge_normal(P1, P2):

    t = np.clip(np.random.normal(0.5, 1/6), 0, 1)
    return (1 - t) * P1 + t * P2

def generate_random_points_between_two_trapezoids(FOV, near1, far1, near2, far2, offset1=(0, 0), offset2=(0, 0)):
    """
    在兩個視錐梯形的任意邊上各生成一個隨機點，作為起點和終點。
    """
    # 生成兩個視錐梯形的頂點
    P1_1, P2_1, P3_1, P4_1 = generate_trapezoid_vertices(FOV, near1, far1, *offset1)
    P1_2, P2_2, P3_2, P4_2 = generate_trapezoid_vertices(FOV, near2, far2, *offset2)

    # 定義兩個梯形的邊
    edges1 = [(P1_1, P2_1), (P2_1, P4_1), (P4_1, P3_1), (P3_1, P1_1)]
    # print(edges1)
    edges2 = [(P1_2, P2_2), (P2_2, P4_2), (P4_2, P3_2), (P3_2, P1_2)]

    # 隨機選擇邊並生成點
    random_edge1_index = np.random.choice(len(edges1))
    while random_edge1_index == 0:
        random_edge1_index = np.random.choice(len(edges1))
    # print(random_edge1_index)
    edge1 = edges1[random_edge1_index]
    random_edge2_index = np.random.choice(len(edges2))
    while random_edge2_index == random_edge1_index:
        random_edge2_index = np.random.choice(len(edges2))
    # print(random_edge2_index)
    edge2 = edges2[random_edge2_index]

    point1 = generate_random_point_on_edge(*edge1)
    point2 = generate_random_point_on_edge(*edge2)
    # point1 = generate_random_point_on_edge_normal(*edge1)
    # point2 = generate_random_point_on_edge_normal(*edge2)

    return point1, point2

import matplotlib.pyplot as plt

def calculate_heading(point1, point2):
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    heading_rad = np.arctan2(delta_y, delta_x)
    heading_deg = np.degrees(heading_rad)
    heading_deg = (90 - heading_deg) % 360  # 以北為0度
    return heading_deg

def get_random_points_between_two_trapezoids(FOV, near1, far1, near2, far2, offset1, offset2):
    point1, point2 = generate_random_points_between_two_trapezoids(FOV, near1, far1, near2, far2, offset1, offset2)
    heading = calculate_heading(point1, point2)
    return heading, point1, point2


def plot_two_trapezoids_with_random_points(point1, point2, trapezoid1, trapezoid2):

    heading = calculate_heading(point1, point2)
    # return heading, point1, point2

    print(f"Point1: {point1}")
    print(f"Point2: {point2}")
    print(f"Heading from point1 to point2: {heading:.2f} degrees")

    
    # 梯形1
    plt.plot(
        [trapezoid1[0][0], trapezoid1[1][0], trapezoid1[3][0], trapezoid1[2][0], trapezoid1[0][0]],
        [trapezoid1[0][1], trapezoid1[1][1], trapezoid1[3][1], trapezoid1[2][1], trapezoid1[0][1]],
        'b-', label="Trapezoid 1"
    )

    # 梯形2
    plt.plot(
        [trapezoid2[0][0], trapezoid2[1][0], trapezoid2[3][0], trapezoid2[2][0], trapezoid2[0][0]],
        [trapezoid2[0][1], trapezoid2[1][1], trapezoid2[3][1], trapezoid2[2][1], trapezoid2[0][1]],
        'g-', label="Trapezoid 2"
    )

    # 繪製隨機點
    plt.scatter([point1[0], point2[0]], [point1[1], point2[1]], c='red', label="Random Points")
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r--', label="Random Path")

    # 標注座標值
    plt.text(point1[0], point1[1], f"({point1[0]:.2f}, {point1[1]:.2f})", fontsize=12, ha='right')
    plt.text(point2[0], point2[1], f"({point2[0]:.2f}, {point2[1]:.2f})", fontsize=12, ha='right')

    # 標注點一或點二
    plt.annotate('Point 1', (point1[0], point1[1]), textcoords="offset points", xytext=(-10,-10), ha='center', fontsize=12, color='blue')
    plt.annotate('Point 2', (point2[0], point2[1]), textcoords="offset points", xytext=(-10,-10), ha='center', fontsize=12, color='blue')

    # 設置
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.title("Random Points Between Two Trapezoids")
    plt.show()


# FOV = 60
# near1, far1 = 50, 1600
# near2, far2 = 50, 1600 
# offset1 = (0, 0)  # 梯形1位置
# offset2 = (0, 550)  # 梯形2位置

# plot_two_trapezoids_with_random_points(FOV, near1, far1, near2, far2, offset1, offset2)
