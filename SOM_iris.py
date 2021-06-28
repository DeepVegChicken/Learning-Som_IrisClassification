"""  基于SOM算法的iris数据集分类 """
import math
import numpy as np
from minisom import MiniSom
from sklearn import datasets
from numpy import sum as npsum
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec


# 分类函数
def classify(som,data,winmap):
    default_class = npsum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


# 可视化
def show(som):
    """ 在输出层画标签图案 """
    plt.figure(figsize=(16, 16))
    # 定义不同标签的图案标记
    markers = ['o', 's', 'D']
    colors = ['C0', 'C1', 'C2']
    category_color = {'setosa': 'C0', 'versicolor': 'C1', 'virginica': 'C2'}

    # 背景上画U-Matrix
    heatmap = som.distance_map()
    # 画背景图
    plt.pcolor(heatmap, cmap='bone_r')

    for cnt, xx in enumerate(X_train):
        w = som.winner(xx)
        # 在样本Heat的地方画上标记
        plt.plot(w[0] + .5, w[1] + .5, markers[Y_train[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[Y_train[cnt]], markersize=12, markeredgewidth=2)

    plt.axis([0, size, 0, size])
    ax = plt.gca()
    # 颠倒y轴方向
    ax.invert_yaxis()
    legend_elements = [Patch(facecolor=clr, edgecolor='w', label=l) for l, clr in category_color.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))
    plt.show()

    """ 在每个格子里画饼图，且用颜色表示类别，用数字表示总样本数量 """
    plt.figure(figsize=(16, 16))
    the_grid = GridSpec(size, size)

    for position in winmap.keys():
        label_fracs = [winmap[position][label] for label in [0, 1, 2]]
        plt.subplot(the_grid[position[1], position[0]], aspect=1)
        patches, texts = plt.pie(label_fracs)
        plt.text(position[0] / 100, position[1] / 100, str(len(list(winmap[position].elements()))),
                 color='black', fontdict={'weight': 'bold', 'size': 15}, va='center', ha='center')
    plt.legend(patches, class_names, loc='center right', bbox_to_anchor=(-1, 9), ncol=3)
    plt.show()


if __name__ == '__main__':
    # 导入数据集
    iris = datasets.load_iris()
    # 提取iris数据集的标签与数据
    feature_names = iris.feature_names
    class_names = iris.target_names
    X = iris.data
    Y = iris.target
    # 划分训练集、测试集  7:3
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # 样本数量
    N = X_train.shape[0]
    # 维度/特征数量
    M = X_train.shape[1]
    # 最大迭代次数
    max_iter = 200
    # 经验公式：决定输出层尺寸
    size = math.ceil(np.sqrt(5 * np.sqrt(N)))
    print("训练样本个数:{}  测试样本个数:{}".format(N, X_test.shape[0]))
    print("输出网格最佳边长为:", size)

    # 初始化模型
    som = MiniSom(size, size, M, sigma=3, learning_rate=0.5, neighborhood_function='bubble')
    # 初始化权值
    som.pca_weights_init(X_train)
    # 模型训练
    som.train_batch(X_train, max_iter, verbose=False)

    # 利用标签信息，标注训练好的som网络
    winmap = som.labels_map(X_train, Y_train)
    # 进行分类预测
    y_pred = classify(som, X_test, winmap)
    # 展示在测试集上的效果
    print(classification_report(Y_test, np.array(y_pred)))

    # 可视化
    show(som)
