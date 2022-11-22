import numpy as np
import pandas as pd
import math
import random
import copy
import time
import matplotlib.pyplot as plt

CAR_MAX = 250  # 车的容量
ITERATION = 180  # 最大迭代次数
M = 1000000  # 一个很大的数

dt = pd.read_excel("./data.xlsx")  # 复制数据
data_e = np.array(dt.iloc[:, 1:3])  # 单独取出坐标
data_con = np.array(dt.iloc[:, 3])  # 客户点的量  0 - 120  data_con[k]即为第k个点的装载量
MIN_CAR = data_con.sum() // CAR_MAX + 1  # 完成装载最少的车辆数

C = np.zeros([data_e.shape[0], data_e.shape[0]])  # 距离矩阵，创建零矩阵
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        C[i][j] = math.floor(math.sqrt(
            (data_e[i][0] - data_e[j][0]) ** 2 + (
                    data_e[i][1] - data_e[j][1]) ** 2) * 10 ** 3) / 10 ** 3  # 计算并取三位小数

S = np.zeros([data_e.shape[0], data_e.shape[0]])  # 节约矩阵，创建零矩阵
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        S[i][j] = C[i][0] + C[0][j] - C[i][j]  # 节约矩阵
        if i == j:
            S[i][j] = 0

S = np.triu(S)
S_new = np.array([range(S.size), S.flatten()]).T  # [[编号（解码回矩阵），节约矩阵的saving]]
index = np.argsort(S.flatten())[::-1]  # 排序编码下标


def saving(S_new=S_new, index=index, data_con=data_con):
    solve = []
    for i in range(len(data_con) - 1):  # 初始化
        solve.append([[0, i + 1, 0], data_con[i + 1]])  # solve[k][0]第k+1个客户点,solve[k][1]现在的装载值   1 - 70

    can_use_left = [i + 1 for i in range(len(data_con) - 1)]  # 左边为0的点  1 - 70
    can_use_right = [i + 1 for i in range(len(data_con) - 1)]  # 右边为0的点  1 - 70

    k = 0
    while S_new[index[k]][1] > 0:  # 第k+1大的节约值是否大于0
        x, y = int(S_new[index[k]][0] // S.shape[0]), int(S_new[index[k]][0] % S.shape[0])  # x,y为第x个客户和第y个客户
        if (x in can_use_right) and (y in can_use_left):  # x-0-0-y
            index_x, index_y = 0, 0
            for i in range(len(solve)):
                if x in solve[i][0]:
                    index_x = i
                    break
            for i in range(len(solve)):
                if y in solve[i][0]:
                    index_y = i
                    break

            if (index_x != index_y) and (solve[index_x][1] + solve[index_y][1] < CAR_MAX):  # 确定可以合并
                can_use_left.remove(y)  # 删去x,y的各自的左、右0
                can_use_right.remove(x)
                solve[index_x][0].pop()  # 删去解上的0
                solve[index_y][0].pop(0)
                solve[index_x][0].extend(solve[index_y][0])  # 两列表合并
                solve[index_x][1] += solve[index_y][1]  # 解合并
                del solve[index_y]  # 删元素

        elif (y in can_use_right) and (x in can_use_left):  # y-0-0-x
            index_x, index_y = 0, 0
            for i in range(len(solve)):
                if x in solve[i][0]:
                    index_x = i
                    break
            for i in range(len(solve)):
                if y in solve[i][0]:
                    index_y = i
                    break

            if (index_x != index_y) and (solve[index_x][1] + solve[index_y][1] < CAR_MAX):  # 确定可以合并
                can_use_left.remove(x)  # 删去x,y的各自的左、右0
                can_use_right.remove(y)
                solve[index_y][0].pop()  # 删去解上的0
                solve[index_x][0].pop(0)
                solve[index_y][0].extend(solve[index_x][0])  # 两列表合并
                solve[index_y][1] += solve[index_x][1]  # 解合并
                del solve[index_x]  # 删元素

        elif (x in can_use_left) and (y in can_use_left):
            index_x, index_y = 0, 0
            for i in range(len(solve)):
                if x in solve[i][0]:
                    index_x = i
                    break
            for i in range(len(solve)):
                if y in solve[i][0]:
                    index_y = i
                    break

            if (index_x != index_y) and (solve[index_x][1] + solve[index_y][1] < CAR_MAX):
                can_use_left.remove(x)
                can_use_left.remove(y)
                solve[index_x][0].pop(0)
                solve[index_y][0].pop(0)
                c = solve[index_y][0][-2]
                can_use_right.remove(c)
                can_use_left.append(c)
                solve[index_y][0] = solve[index_y][0][::-1]
                solve[index_y][0].extend(solve[index_x][0])
                solve[index_y][1] += solve[index_x][1]
                del solve[index_x]

        elif (x in can_use_right) and (y in can_use_right):
            index_x, index_y = 0, 0
            for i in range(len(solve)):
                if x in solve[i][0]:
                    index_x = i
                    break
            for i in range(len(solve)):
                if y in solve[i][0]:
                    index_y = i
                    break

            if (index_x != index_y) and (solve[index_x][1] + solve[index_y][1] < CAR_MAX):
                can_use_right.remove(x)
                can_use_right.remove(y)
                solve[index_x][0].pop()
                solve[index_y][0].pop()
                c = solve[index_y][0][1]
                can_use_left.remove(c)
                can_use_right.append(c)
                solve[index_y][0] = solve[index_y][0][::-1]
                solve[index_x][0].extend(solve[index_y][0])
                solve[index_x][1] += solve[index_y][1]
                del solve[index_y]

        k += 1

    return solve


def desolve(solve, C=C):
    """
    求解解的总花费
    :param solve: 解
    :param C: 距离矩阵
    :return: 总距离total_distance
    """
    total_use = len(solve) * 100
    total_distance = 0
    for i in range(len(solve)):
        row_distance = 0
        for j in range(len(solve[i][0]) - 1):
            row_distance = row_distance + C[solve[i][0][j]][solve[i][0][j + 1]]
        solve[i].append(row_distance)
        total_distance += row_distance
    total_use += total_distance

    return total_use, total_distance, solve


def deal_solve(solve_k, data_e=data_e):
    """
    处理每一个子解的序列，使之方便绘图
    :param solve_k: 每一个子解
    :param data_e: 坐标
    :return: 绘图用x,y
    """
    decode_x, decode_y = data_e[solve_k[0], 0], data_e[solve_k[0], 1]
    x, y = [], []
    for ix in range(len(decode_x) - 1):
        x.append([decode_x[ix], decode_x[ix + 1]])
        y.append([decode_y[ix], decode_y[ix + 1]])
    return x, y


def solve_plot(solve, data_e=data_e):
    """
    对输入的解作vrp图
    :param solve:解
    :param data_e:坐标
    :return:图
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("solve")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.scatter(data_e[:, 0], data_e[:, 1], c='r', marker='.')  # 绘制初步的散点

    for i, txt in enumerate(range(data_e.shape[0])):  # 给点写上编号
        ax1.annotate(txt, (data_e[i, 0], data_e[i, 1]))

    for i in range(len(solve)):
        x, y = deal_solve(solve[i])
        for j in range(len(x)):
            plt.plot(x[j], y[j], color='r')

    plt.show()


def initial():
    solve_greedy = saving()
    total_use, total_distance, initial_solve = desolve(solve_greedy)
    solve_plot(initial_solve)
    return total_use, total_distance, initial_solve


def solve_to_list(solve):
    solve_new = copy.deepcopy(solve)
    now_list = []
    for each in solve_new:
        each[0].pop()
        each[0].pop(0)
        now_list.extend(each[0])
    return now_list


def list_to_solve(new, datacon=data_con):
    solve = []
    A = np.full(len(new) * len(new), M, dtype=int).reshape(len(new), len(new))
    for i in range(len(new)):
        for j in range(i, len(new)):
            vec = copy.deepcopy(new[i:j + 1])
            if sum(datacon[vec]) <= CAR_MAX:
                vec.insert(0, 0)
                vec.insert(len(vec), 0)
                A[i, j] = 0
                for x in range(len(vec) - 1):
                    A[i, j] += C[vec[x], vec[x + 1]]

    vec = [A[0][0]]
    v = np.zeros(len(new))
    for i in range(len(new) - 1):
        vec.append(vec[i] + A[i + 1][i + 1])

    for j in range(len(new) - 1):
        i = j
        while A[i][j + 1] != M:
            if vec[j + 1] > A[i + 1][j + 1] + vec[i]:
                vec[j + 1] = A[i + 1][j + 1] + vec[i]
                v[j + 1] = i

            if i != 0 and A[i][j + 1] != M:
                i -= 1
            else:
                break

    k = len(new)
    to_find = []
    while k != 0:
        to_find.insert(0, [v[int(k - 1)], int(k - 1)])
        k = v[int(k - 1)]

    for each in to_find:
        new_in = copy.deepcopy(new[int(each[0]):int(each[1] + 1)])
        new_in.append(0)
        new_in.insert(0, 0)
        solve.append([new_in, sum(datacon[new[int(each[0]):int(each[1] + 1)]])])

    use, distance, solve = desolve(solve)
    return use, distance, solve


def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length > 0 else 0
    random_list = []
    while len(random_list) < length:
        new = random.randint(start, stop)
        if new not in random_list:
            random_list.append(new)

    return random_list


def SA(new, optimal, T):
    prob = math.exp((optimal - new) / T)
    yi = random.random()
    return prob > yi


def SA_optimal(datacon=data_con):
    """
    模拟退火算法
    :param datacon: 装载量
    :return: 优化解
    """
    optimal_distance_vector = []

    arr = np.arange(1, len(datacon))
    np.random.shuffle(arr)
    arr = list(arr)
    optimal_use, optimal_distance, optimal_solve = list_to_solve(arr)
    optimal_list = copy.deepcopy(arr)

    T = 110  # 温度
    ITERATION_IN = 100  # 内部循环
    delta_T = T / ITERATION
    random_num = 5

    new_list = copy.deepcopy(optimal_list)
    for i in range(ITERATION):
        print(i)
        for j in range(ITERATION_IN):
            change_point = np.array(random_int_list(0, len(datacon) - 2, random_num))
            new_get = []
            for x in change_point:
                new_get.append(copy.deepcopy(new_list[x]))
                new_list[x] = M

            new_point = np.random.permutation(change_point)
            for k in range(random_num):
                for l in range(random_num):
                    if change_point[k] == new_point[l]:
                        new_list[change_point[k]] = new_get[l]

            new_use, new_distance, new_solve = list_to_solve(new_list)

            if new_use < optimal_use:
                optimal_solve = new_solve
                optimal_use = new_use
                optimal_distance = new_distance
            elif new_use > optimal_use and SA(new_use, optimal_use, T):
                optimal_solve = new_solve
                optimal_use = new_use
                optimal_distance = new_distance

        T -= delta_T

    return optimal_use, optimal_distance, optimal_solve, optimal_distance_vector


# def test_optimal_solve(optimal_solve, out_k, in_k, out_point, in_point):
#     """
#     更新距离用
#     :param optimal_solve:原解
#     :param out_k:取元素的子解
#     :param in_k:放元素的子解
#     :param out_point:取元素的位置
#     :param in_point:放元素的位置
#     :return: 新解，已更新距离
#     """
#     test = copy.deepcopy(optimal_solve)
#
#     test[in_k][1] += data_con[test[out_k][0][out_point]]  # 更新装载量
#     test[out_k][1] -= data_con[test[out_k][0][out_point]]
#
#     test[in_k][0].insert(in_point, test[out_k][0][out_point])  # 增删元素
#     del test[out_k][0][out_point]
#
#     for c in test:  # 删去最后的距离量
#         c.pop()
#
#     return test


# def optimal(total_distance, initial_solve, datacon=data_con):
#     """
#     优化算法
#     :param total_distance: 初始化的距离值
#     :param initial_solve: 初始解
#     :param datacon: 装载量
#     :return: 优化解
#     """
#     optimal_solve = copy.deepcopy(initial_solve)
#     optimal_distance = copy.deepcopy(total_distance)
#     optimal_distance_vector = []
#     for i in range(ITERATION):
#         solve_out_index, solve_in_index = random.sample(range(len(optimal_solve)), 2)  # 选某两个子解
#         out_point_index, in_point_index = random.randint(1, len(optimal_solve[solve_out_index][0]) - 2), random.randint(
#             1, len(optimal_solve[solve_in_index][0]) - 1)  # 选择这两个解的取出和放置点
#
#         if datacon[optimal_solve[solve_out_index][0][out_point_index]] + optimal_solve[solve_in_index][1] < CAR_MAX:
#
#             test = test_optimal_solve(optimal_solve, solve_out_index, solve_in_index, out_point_index,
#                                       in_point_index)
#
#             test_distance, test = desolve(test)  # 总路程，重新生成有距离的矩阵
#             if test_distance < optimal_distance:
#                 optimal_distance = test_distance
#                 optimal_solve = test
#         i += 1
#         optimal_distance_vector.append(optimal_distance)
#     return optimal_distance, optimal_solve, optimal_distance_vector


def main():
    start = time.time()
    total_use, total_distance, initial_solve = initial()
    optimal_use, optimal_distance, optimal_solve, optimal_distance_vector = SA_optimal()

    # solve_plot(optimal_solve)  # 绘制vrp图
    # plt.plot(range(ITERATION), optimal_distance_vector)  # 绘制迭代图
    # plt.show()

    end = time.time()
    print("总迭代数为", ITERATION, "\n", "节约算法花费为", total_use, "节约算法路径", total_distance, "\n"
                                                                                                      "优化花费为",
          optimal_use, "优化路径",
          optimal_distance, "\n"
                            "耗费时间为", end - start, "秒，", "平均耗时", (end - start) / ITERATION, "秒。")
    print("saving算法的解：", end="\n")
    for i in initial_solve:
        print(i, end="\n")
    print("{:=^80s}".format("Split Line"))
    print("optimal saving 算法解：", end="\n")
    for j in optimal_solve:
        print(j, end="\n")
    return optimal_distance, optimal_solve


main()
