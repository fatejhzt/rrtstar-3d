import math
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D    #这里只是用Axes3D函数，所以只导入了Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
class RRT:
    class Node:  # 创建节点
        def __init__(self, x, y,z):
            self.x = x  # 节点坐标
            self.y = y
            self.z = z
            self.cost = 0.0
            self.path_x = []  # 路径，作为画图的数据
            self.path_y = []
            self.path_z = []
            self.parent = None  # 父节点

    class AreaBounds:
        """区域大小"""
        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

            self.zmin = float(area[4])
            self.zmax = float(area[5])

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=0.01,  # 树枝长度
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.0,
                 ):
        """
        Setting Parameter
        start:起点 [x,y]
        goal:目标点 [x,y]
        obstacleList:障碍物位置列表 [[x,y,size],...]
        rand_area: 采样区域 x,y，z ∈ [min,max，]
        play_area: 约束随机树的范围 [xmin,xmax,ymin,ymax]
        robot_radius: 机器人半径
        expand_dis: 扩展的步长
        goal_sample_rate: 采样目标点的概率，百分制.default: 5，即表示5%的概率直接采样目标点
        """
        self.start = self.Node(start[0],start[1],start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])  # 终点(6,10)
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]

        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)  # 树枝生长区域，左下(-2,0)==>右上(12,14)
        else:
            self.play_area = None  # 数值无限生长

        self.expand_dis = expand_dis  #树枝一次的生长长度
        self.goal_sample_rate = goal_sample_rate#多少概率选择终点
        self.max_iter = max_iter#最大迭代次数
        self.obstacle_list = obstacle_list #障碍物的坐标
        self.node_list = [] #保存节点
        self.robot_radius = robot_radius #随机点的搜索半径

#路径规划
    def planning(self,animation=True,camera=None):

        #将点作为根节点x_{init}，加入到随机树的节点集合中。
        self.node_list = [self.start]
        for i in range(self.max_iter):
            #从可行区域内随机选取一个节点x_{rand}
            rnd_node = self.sample_free()

            # 已生成的树中利用欧氏距离判断距离x_{rand}最近的点x_{near}。
            # 从已知节点中选择和目标节点最近的节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)  # 最接近的节点的索引
            nearest_node = self.node_list[nearest_ind]  # 获取该最近已知节点的坐标

            #从x_{near} 与 x_{rand} 的连线方向上扩展固定步长 u，得到新节点 x_{new}
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 如果在可行区域内，且x_{near}与x_{new}之间无障碍物
            # 判断新点是否在规定的树的生长区域内，新点和最近点之间是否存在障碍物
            if self.is_inside_play_area(new_node, self.play_area) and \
                    self.obstacle_free(new_node, self.obstacle_list, self.robot_radius):
                # 都满足才保存该点作为树节点\

                # 得到范围中点的索引
                nearInds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, nearInds)

                self.node_list.append(new_node)

                self.rewire(new_node, nearInds)


            # 如果此时得到的节点x_new到目标点的距离小于扩展步长，则直接将目标点作为x_rand。
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y,self.node_list[-1].z) <= self.expand_dis:
                # 以新点为起点，向终点画树枝
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                # 如果最新点和终点之间没有障碍物True
                if self.obstacle_free(final_node, self.obstacle_list, self.robot_radius):
                    # 返回最终路径
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node, camera)
        return None




    # 距离最近的已知节点坐标，随机坐标，从已知节点向随机节点的延展的长度
    def steer(self,from_node,to_node,extend_lengh=float("inf")):
        # d已知点和随机点之间的距离，theta两个点之间的夹角
        d,theta_xy,theta_z = self.calc_distance_and_angle(from_node,to_node)

        #如果$x_{near}$与$x_{rand}$间的距离小于步长，则直接将$x_{rand}$作为新节点$x_{new}$
        if extend_lengh >= d: #如果树枝的生长长度超出随机点，就用随机点位置作为新的节点
            new_x = to_node.x
            new_y = to_node.y
            new_z = to_node.z
        else:
            new_x = from_node.x + math.cos(theta_xy) * extend_lengh  # 最近点 x + cos * extend_len
            new_y = from_node.y + math.sin(theta_xy) * extend_lengh  # 最近点 y + sin * extend_len
            new_z = from_node.z + math.sin(theta_z) * extend_lengh  # 最近点 z + sin * extend_len

        new_node = self.Node(new_x,new_y,new_z)
        new_node.path_x = [from_node.x]  #最近点
        new_node.path_y = [from_node.y]
        new_node.path_z = [from_node.z]

        new_node.path_x.append(new_x)
        new_node.path_y.append(new_y)
        new_node.path_z.append(new_z)

        new_node.cost += self.expand_dis
        new_node.parent = from_node  # 根节点变成最近点，用来指明方向
        #将根节点进行更新，将将更节点修改为最近点，用来指明方向
        return new_node
    #
    # def generate_final_course(self,goal_ind):
    #     """生成路径
    #     Args:
    #         goal_ind (_type_): 目标点索引
    #     Returns:
    #         _type_: _description_
    #     """
    #     path = [[self.end.x,self.end.y,self.end.z]]  #将终点保存
    #     node = self.node_list[goal_ind]
    #     while node.parent is not None:
    #         path.append([node.x,node.y,node.z])
    #         node = node.parent
    #     path.append([node.x,node.y,node.z])
    #     return path
    def generate_final_course(self, lastIndex):
        path = [[self.end.x, self.end.y, self.end.z]]

        while lastIndex is not None and isinstance(lastIndex, int) and lastIndex >= 0 and lastIndex < len(
                self.node_list):
            node = self.node_list[lastIndex]
            path.append([node.x, node.y, node.z])
            lastIndex = node.parent

        path.append([self.start.x, self.start.y, self.start.z])
        return path

    #在三维空间中计算点与点的距离
    def calc_dist_to_goal(self, x, y, z):
        """计算(x, y, z)离目标点的距离
        """
        dx = x - self.end.x
        dy = y - self.end.y
        dz = z - self.end.z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


    def sample_free(self):
        # 以（100-goal_sample_rate）%的概率随机生长，(goal_sample_rate)%的概率朝向目标点生长
        if random.randint(0,100)>self.goal_sample_rate:  #大于5%就不选终点方向作为下个节点
            rnd = self.Node(
                random.uniform(self.min_rand,self.max_rand),  #在树枝生长区域中随机取一个点
                random.uniform(self.min_rand,self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = self.Node(self.end.x,self.end.y,self.end.z)
        return rnd


    def draw_graph(self,rnd=None,camera=None):
        if camera == None:
            plt.clf()
        #use the key of esc to stop simulation
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event:[exit() if event.key == 'escape' else None])

        fig = plt.figure(1)
        # ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')
        # 画随机点
        if rnd is not None:
            ax.scatter(rnd.x, rnd.y, rnd.z, marker='^', color='k')
            if self.robot_radius > 0.0:
                self.plot_sphere(ax, rnd.x, rnd.y, rnd.z, self.robot_radius, color='r')

        #Drew generated tree in 3D
        for node in self.node_list:
            if node.parent:
                ax.plot( node.path_x, node.path_y,node.path_z,color="green")

        # drew obstacles in 3D
        for (ox,oy,oz,size) in self.obstacle_list:
            self.plot_sphere(ax,ox,oy,oz,size)

        #if a feasible area is defined,drew it in 3D
        if self.play_area is not None:
            """ 
                    绘制正方形框
            """
            vertices = [
                (self.play_area.xmin, self.play_area.ymin, self.play_area.zmin),
                (self.play_area.xmax,self.play_area.ymin, self.play_area.zmin),
                (self.play_area.xmax, self.play_area.ymax, self.play_area.zmin),
                (self.play_area.xmin, self.play_area.ymax, self.play_area.zmin),
                (self.play_area.xmin, self.play_area.ymin, self.play_area.zmax),
                (self.play_area.xmax, self.play_area.ymin, self.play_area.zmax),
                (self.play_area.xmax, self.play_area.ymax, self.play_area.zmax),
                (self.play_area.xmin, self.play_area.ymax, self.play_area.zmax),
            ]
            # 定义顶点连接
            edges = [
                [vertices[0], vertices[1], vertices[2], vertices[3], vertices[0]],
                [vertices[4], vertices[5], vertices[6], vertices[7], vertices[4]],
                [vertices[0], vertices[4]],
                [vertices[1], vertices[5]],
                [vertices[2], vertices[6]],
                [vertices[3], vertices[7]],
            ]

            # 绘制线
            for edge in edges:
                x, y, z = zip(*edge)
                ax.plot(x, y, z, color='black')

            # 绘制透明的正方体表面
            ax.add_collection3d(Poly3DCollection([edges[0]], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.1))
            ax.add_collection3d(Poly3DCollection([edges[1]], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.1))

        ax.scatter(self.start.x, self.start.y, self.start.z, color='r', marker='x')
        ax.scatter(self.end.x, self.end.y, self.end.z, color='r', marker='x')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.grid(True)
        plt.pause(0.01)

        if camera is not None:
            camera.snap()

    def choose_parent(self,newNode,nearInds):
        if len(nearInds)==0:
            return newNode

        dList = []
        for i in nearInds:
            dx = newNode.x - self.node_list[i].x
            dy = newNode.y - self.node_list[i].y
            dz = newNode.z - self.node_list[i].z

            d = math.sqrt(dx**2 + dy**2 + dz**2)
            theta_xy = math.atan2(dy, dx)
            theta_z = math.atan2(dz, math.sqrt(dx**2 + dy**2))

            if self.check_collision(self.node_list[i], theta_xy, theta_z, d):
                dList.append(self.node_list[i].cost + d)
            else:
                dList.append(float('inf'))

        minCost = min(dList)
        minInd = nearInds[dList.index(minCost)]

        if minCost == float('inf'):
            print("min cost is inf")
            return newNode

        newNode.cost = minCost
        newNode.parent = minInd

        return newNode

    # 重新
    def find_near_nodes(self,newNode):
        n_node = len(self.node_list)
        r = 1.0*math.sqrt((math.log(n_node))/n_node)
        d_list = [(node.x - newNode.x)**2 + (node.y - newNode.y)**2 +(node.z - newNode.z)**2
                  for node in self.node_list]
        near_inds = [d_list.index(i) for i in d_list if i <= r**2]
        return  near_inds

    #画出障碍物
    @staticmethod
    def plot_sphere(ax, x, y, z, radius, color="-b", resolution=100):
        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        phi, theta = np.meshgrid(phi, theta)
        x_sphere = x + radius * np.sin(phi) * np.cos(theta)
        y_sphere = y + radius * np.sin(phi) * np.sin(theta)
        z_sphere = z + radius * np.cos(phi)
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.3)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        # 计算所有已知节点和随机节点之间的距离
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 + (node.z - rnd_node.z) ** 2
                 for node in node_list]
        # 获得距离最小的节点的索引
        minind = dlist.index(min(dlist))

        return minind

    #判断选择的点是否在可行域
    @staticmethod
    def is_inside_play_area(node, play_area):
        if play_area is None:
            return True  # 如果没有定义可行区域，那么任何位置都是合适的

        if node.x < play_area.xmin or node.x > play_area.xmax or \
                node.y < play_area.ymin or node.y > play_area.ymax or \
                node.z < play_area.zmin or node.z > play_area.zmax:
            return False  # 如果节点的 x、y 或 z 坐标在可行区域外，返回 False（不合适）
        else:
            return True  # 如果节点的 x、y 和 z 坐标在可行区域内，返回 True（合适）


    #该函数的作用是判断p_new点和p_now的连线是否碰撞到障碍物
    @staticmethod
    def obstacle_free(node, obstacleList, robot_radius):  # 目标点，障碍物中点和半径，移动时的占地半径

        if node is None:
            return False

        for (ox, oy, oz, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            dz_list = [oz - z for z in node.path_z]
            d_list = [dx * dx + dy * dy + dz * dz for (dx, dy, dz) in zip(dx_list, dy_list, dz_list)]

            if min(d_list) <= (size + robot_radius) ** 2:
                return False  # collision

        return True  # safe

    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        # Return minimum distance between line segment vw and point p
        if np.array_equal(v, w):
            return np.linalg.norm(p - v) ** 2  # v == w case
        l2 = np.linalg.norm(w - v) ** 2  # i.e. |w-v|^2 -  avoid a sqrt
        # Consider the line extending the segment,
        # parameterized as v + t (w - v).
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment vw.
        t = max(0, min(1, np.dot(p - v, w - v) / l2))
        projection = v + t * (w - v)  # Projection falls on the segment
        return np.linalg.norm(p - projection) ** 2

    def check_segment_collision(self, x1, y1, z1, x2, y2, z2):
        for (ox, oy, oz, size) in self.obstacle_list:
            dd = self.distance_squared_point_to_segment(
                np.array([x1, y1, z1]),
                np.array([x2, y2, z2]),
                np.array([ox, oy, oz]))
            if dd <= size ** 2:
                return False  # collision
        return True

    def check_collision(self, nearNode, theta_xy, theta_z, d):
        tmpNode = copy.deepcopy(nearNode)

        end_x = tmpNode.x + math.cos(theta_xy) * math.cos(theta_z) * d
        end_y = tmpNode.y + math.sin(theta_xy) * math.cos(theta_z) * d
        end_z = tmpNode.z + math.sin(theta_z) * d

        return self.check_segment_collision(tmpNode.x, tmpNode.y, tmpNode.z, end_x, end_y, end_z)


    def rewire(self, newNode, nearInds):
        n_node = len(self.node_list)
        for i in nearInds:
            nearNode = self.node_list[i]

            d = math.sqrt((nearNode.x - newNode.x) ** 2
                          + (nearNode.y - newNode.y) ** 2
                          + (nearNode.z - newNode.z) ** 2)

            s_cost = newNode.cost + d

            if nearNode.cost > s_cost:
                theta_xy = math.atan2(newNode.y - nearNode.y, newNode.x - nearNode.x)
                theta_z = math.atan2(newNode.z - nearNode.z, math.sqrt((newNode.x - nearNode.x) ** 2 + (newNode.y - nearNode.y) ** 2))

                if self.check_collision(nearNode, theta_xy, theta_z, d):
                    nearNode.parent = n_node - 1
                    nearNode.cost = s_cost
            return nearNode
    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """计算两个节点间的距离和方位角
        Args:
            from_node: 起始节点
            to_node: 目标节点
        Returns:
            d: 两节点之间的直线距离
            theta_xy: 从起始节点到目标节点的水平方位角（弧度）
            theta_z: 从起始节点到目标节点的垂直方位角（弧度）
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # 计算两节点之间的直线距离
        theta_xy = math.atan2(dy, dx)  # 计算水平方位角的弧度值
        theta_z = math.atan2(dz, math.sqrt(dx ** 2 + dy ** 2))  # 计算垂直方位角的弧度值
        return d, theta_xy, theta_z

def main(gx=6.0, gy=10.0,gz=10.0):
    print("start " + __file__)
    fig = plt.figure(1)

    # camera = Camera(fig) # 保存动图时使用
    camera = None  # 不保存动图时，camara为None
    show_animation = True

    # ====Search Path with RRT====
    obstacleList = [[0.1, 0.2, 0.1, 0.1],[0.3,0.1,0.5,0.1],[0.2,0.1,0.5,0.1]]  # [x, y, radius]

    # Set Initial parameters

    rrt = RRT(
        start=[0.1, 0.0, 0.3],  # 起点位置
        goal=[ 0.6, 0.2, 0.5],  # 终点位置
        rand_area=[0.0, 1.0],  # 树枝可生长区域[xmin, xmax]
        obstacle_list=obstacleList,  # 障碍物
        play_area=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],  # 树的生长区域，左下[-2, 0, 0] ==> 右上[13, 13, 13]
        robot_radius=0.01  # 搜索半径
    )

    path = rrt.planning(animation=show_animation, camera=camera)
    print(np.array(path).size)
    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # 绘制最终路径
        if show_animation:
            rrt.draw_graph(camera=camera)  # 三维绘图
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')  # 添加三维子图
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_zlim([0, 1])

            # 绘制RRT的生成过程
            for node in rrt.node_list:
                if node.parent:
                    ax.plot(node.path_x, node.path_y, node.path_z, "-g", alpha=0.5)  # 使用 alpha 控制透明度
            # 绘制最终路径
            ax.plot([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], '-r')
            plt.show()



if __name__ == '__main__':
    main()