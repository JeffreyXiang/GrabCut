import numpy as np
import imageio
import colorsys
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import maxflow

fig = plt.figure()
image = (imageio.imread('test.png') / 255.)[..., :3]
mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.int8)
cluster = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.int8)
seg = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.int8)

GMM_num = 5
GMM_weight = np.zeros((2, GMM_num))
GMM_mean = np.zeros((2, GMM_num, 3))
GMM_cov = np.zeros((2, GMM_num, 3, 3))
GMM_cov_det = np.zeros((2, GMM_num))
GMM_cov_inv = np.zeros((2, GMM_num, 3, 3))

colorBar = np.array([colorsys.hsv_to_rgb(i, 1, 1) for i in np.random.permutation(np.linspace(0, 1, 2 * GMM_num))])

isMousePressed = False
mousePosStart = None
mousePosEnd = None
mousePos = None

render = ''
status = 'start'

def energy_D(z, alpha, k):
    offset = (z - GMM_mean[alpha, k])[:, :, :, None]
    offset_T = (z - GMM_mean[alpha, k])[:, :, None, :]
    res = -np.log(GMM_weight[alpha, k]) + 0.5 * np.log(GMM_cov_det[alpha, k]) + (0.5 * offset_T @ GMM_cov_inv[alpha, k] @ offset).squeeze()
    return res

def initGMM():
    sum_0 = (seg[:, :, 0] == 0).sum()
    sum_1 = (seg[:, :, 0] == 1).sum()
    for i in range(GMM_num):
        if (sum_0 > 0):
            flag_0 = (seg[:, :, 0] == 0) * (cluster[:, :, 0] == i)
            GMM_weight[0, i] = flag_0.sum() / sum_0
            if GMM_weight[0, i] > 0:
                pixels_0 = image[flag_0]
                GMM_mean[0, i] = pixels_0.mean(axis=0)
                GMM_cov[0, i] = np.cov(pixels_0, rowvar=False)
                GMM_cov_inv[0, i] = np.linalg.inv(GMM_cov[0, i])
                GMM_cov_det[0, i] = np.linalg.det(GMM_cov[0, i])
        if (sum_1 > 0):
            flag_1 = (seg[:, :, 0] == 1) * (cluster[:, :, 1] == i)
            GMM_weight[1, i] = flag_1.sum() / sum_1
            if GMM_weight[1, i] > 0:
                pixels_1 = image[flag_1]
                GMM_mean[1, i] = pixels_1.mean(axis=0)
                GMM_cov[1, i] = np.cov(pixels_1, rowvar=False)
                GMM_cov_inv[1, i] = np.linalg.inv(GMM_cov[1, i])
                GMM_cov_det[1, i] = np.linalg.det(GMM_cov[1, i])

def assignGMM():
    energy_D_0 = []
    energy_D_1 = []
    for k in range(GMM_num):
        energy_D_0.append(energy_D(image, 0, np.ones(image.shape[:-1], dtype=int) * k))
        energy_D_1.append(energy_D(image, 1, np.ones(image.shape[:-1], dtype=int) * k))
    cluster[:, :, 0] = np.argmin(np.stack(energy_D_0), axis=0)
    cluster[:, :, 1] = np.argmin(np.stack(energy_D_1), axis=0)

def learnGMM():
    sum_0 = ((seg[:, :, 0] == 0) * (mask[:, :, 0] == 2)).sum()
    sum_1 = ((seg[:, :, 0] == 1) * (mask[:, :, 0] == 2)).sum()
    for i in range(GMM_num):
        if (sum_0 > 0):
            flag_0 = (seg[:, :, 0] == 0) * (cluster[:, :, 0] == i)
            GMM_weight[0, i] = flag_0.sum() / sum_0
            if GMM_weight[0, i] > 0:
                pixels_0 = image[flag_0]
                GMM_mean[0, i] = pixels_0.mean(axis=0)
                GMM_cov[0, i] = np.cov(pixels_0, rowvar=False)
                GMM_cov_inv[0, i] = np.linalg.inv(GMM_cov[0, i])
                GMM_cov_det[0, i] = np.linalg.det(GMM_cov[0, i])
        if (sum_1 > 0):
            flag_1 = (seg[:, :, 0] == 1) * (cluster[:, :, 1] == i)
            GMM_weight[1, i] = flag_1.sum() / sum_1
            if GMM_weight[1, i] > 0:
                pixels_1 = image[flag_1]
                GMM_mean[1, i] = pixels_1.mean(axis=0)
                GMM_cov[1, i] = np.cov(pixels_1, rowvar=False)
                GMM_cov_inv[1, i] = np.linalg.inv(GMM_cov[1, i])
                GMM_cov_det[1, i] = np.linalg.det(GMM_cov[1, i])

def onMousePress(event):
    global isMousePressed, mousePosStart, mousePos, mode
    if event.button==1: #鼠标左键点击
        isMousePressed = True
        mousePosStart = (event.xdata, event.ydata)
        mousePos = mousePosStart
        if status == 'start':
            execAction()

def onMouseRelease(event):
    global isMousePressed, mousePosEnd, mousePos, mode
    if event.button==1: #鼠标左键点击
        isMousePressed = False
        mousePosEnd = (event.xdata, event.ydata)
        mousePos = mousePosEnd
        if status == 'initial_seg_start':
            execAction()

def onMouseMotion(event):
    global mousePos
    mousePos = (event.xdata, event.ydata)
    if isMousePressed:
        drawFigure()

def onKeyPress(event):
    global status
    if event.key == 'r' or event.key == 'R':
        status = 'restart'
        execAction()
    if event.key == 'enter':
        if status in ('initial_seg_end', 'init_GMM', 'iteration_cluster', 'iteration_seg'):
            execAction()

def drawFigure():
    plt.clf()
    plt.axis('off')
    render_image = image.copy()
    if 'with_cluster' in render:
        render_cluster = (seg == 0) * cluster[:, :, :1] + (seg == 1) * (cluster[:, :, 1:] + GMM_num)
        render_cluster_image = colorBar[render_cluster.reshape(-1)].reshape(image.shape)
        render_image = render_image * render_cluster_image
    if 'with_seg' in render:
        render_image = 0.5 * render_image * (seg == 1) + 0.5 * render_image
    if 'with_mask' in render:
        render_image = 0.5 * render_image * (mask == 2) + 0.5 * render_image
    plt.imshow(render_image)
    if 'rec_selection' in render:
        plt.plot(
            [mousePosStart[0], mousePosStart[0], mousePos[0], mousePos[0], mousePosStart[0]],
            [mousePosStart[1], mousePos[1], mousePos[1], mousePosStart[1], mousePosStart[1]],
            c = 'r',
        )
    plt.draw()

def execAction():
    global status, render
    if status == 'restart':
        status = 'start'
        render = ''
    elif status == 'start':
        mask.fill(0)
        status = 'initial_seg_start'
        render = 'rec_selection'
    elif status == 'initial_seg_start':
        xl = np.round(min(mousePosStart[0], mousePosEnd[0])).astype(int)
        xh = np.round(max(mousePosStart[0], mousePosEnd[0])).astype(int)
        yl = np.round(min(mousePosStart[1], mousePosEnd[1])).astype(int)
        yh = np.round(max(mousePosStart[1], mousePosEnd[1])).astype(int)
        mask[yl:yh, xl:xh] = 2
        seg[:] = mask[:] // 2
        status = 'initial_seg_end'
        render = 'with_mask'
    elif status == 'initial_seg_end':
        kmeans_B = KMeans(n_clusters=GMM_num).fit(image[mask.squeeze()==0])
        cluster[:, :, 0][mask.squeeze()==0] = kmeans_B.labels_
        kmeans_U = KMeans(n_clusters=GMM_num).fit(image[mask.squeeze()==2])
        cluster[:, :, 1][mask.squeeze()==2] = kmeans_U.labels_
        initGMM()
        status = 'init_GMM'
        render = 'with_cluster'
    elif status == 'init_GMM':
        assignGMM()
        learnGMM()
        status = 'iteration_cluster'
        render = 'with_cluster'
    elif status == 'iteration_cluster':

        status = 'iteration_seg'
        render = 'with_seg'
    elif status == 'iteration_seg':
        assignGMM()
        learnGMM()
        status = 'iteration_cluster'
        render = 'with_cluster'
    drawFigure()
    

if __name__ == '__main__':
    fig.canvas.mpl_connect('key_press_event', onKeyPress)
    fig.canvas.mpl_connect('button_press_event', onMousePress)
    fig.canvas.mpl_connect('button_release_event', onMouseRelease)
    fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)
    drawFigure()
    plt.show()
