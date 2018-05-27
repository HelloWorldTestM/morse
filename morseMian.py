import os
import wave
import matplotlib.pyplot as plt
import numpy as np


## 获取指定文件后缀的函数
def getFileName(path, suffix):
    ''' 获取指定目录下的所有指定后缀的文件名 '''
    filelist = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == suffix:
            filelist.append(i)
    return filelist


## 根据指定时间对音频进行压缩
def getMeanByTime(data, time, params):
    nchannels, sampwidth, framerate, nframes = params[:4]
    ## 划分的帧的数目
    lenPreFrame = int(time * framerate / 1000)
    compressFrame = []
    frameCount = 1
    while (frameCount * lenPreFrame < len(data)):
        tmp = np.mean(np.abs(data[(frameCount - 1) * lenPreFrame:frameCount * lenPreFrame]))
        if (tmp > 0.2):
            compressFrame.append(1)
        else:
            compressFrame.append(0)
        frameCount += 1
    return compressFrame


## 按照k_mains将数据分成多个类型
def k_means(data, type):
    if (len(data) < type):
        print("k_mains error: the number of data less than the kind.")
        return
    lens = len(data)
    # 设置迭代次数
    iters = 10
    # 初始化type个核心
    core_index = np.arange(0, type) * int(lens / type) + int(lens / type / 2)
    data_core = []
    typelist = []
    # 获取type的核心数据
    for index in core_index:
        data_core.append(data[index])
        typelist.append([])

    for it in range(0, iters):
        # 初始化本次迭代的存储分类结果list
        for idx in range(0, type):
            typelist[idx] = []
        for elem in data:
            distance = 100000
            index = 0
            for idx in range(0, type):
                if (np.abs(elem - data_core[idx])) < distance:
                    distance = np.abs(elem - data_core[idx])
                    index = idx
            typelist[index].append(elem)
        # 迭代完之后更新data_core
        for idx in range(0, type):
            data_core[idx] = np.mean(typelist[idx])

    return data_core


# map key是索引，value是梯度值
# 函数：寻找也定梯度值对应的索引
def find_index(map, target):
    keys = map.keys()
    for idx in keys:
        if map[idx] == target:
            map.pop(idx)
            return idx
    return -1


## 按照点的增长速度进行排序，取出前两个
## data需要找到分类点的数据，type是要分的种类
def k_tans(data, type):
    if (len(data) < type):
        print("k_mains error: the number of data less than the kind.")
        return
    lens = len(data)
    gradhelp = []
    gradmap = {}
    for idx in range(1, lens):
        grad = (data[idx] - data[idx - 1]) * 1.0 / data[idx - 1]
        gradhelp.append(grad)
        gradmap[idx] = grad

    gradhelp.sort(reverse=True)
    data_core = []
    for idx in range(0, type - 1):
        # 第idx大的数在gradient中的索引+1
        index = find_index(gradmap, gradhelp[idx])
        data_core.append((data[index] + data[index - 1]) / 2.0)
    data_core.sort()
    return data_core


def translate(list_total, list_flag, data_core_1, data_core_0):
    lens = len(list_total)
    res = ""
    dres = ""
    char_1 = ["*", "-"]
    char_0 = [" ", "   ", "       "]
    char_0_pro = [" ", "a", "awa"]
    for idx in range(0, lens):
        if (list_flag[idx] == 1):
            type = 0
            for core in data_core_1:
                if list_total[idx] < core:
                    break
                type += 1
            res += char_1[type]
            dres += char_1[type]
        else:
            type = 0
            for core in data_core_0:
                if list_total[idx] < core:
                    break
                type += 1
            res += char_0[type]
            dres += char_0_pro[type]
    return dres, res


## 根据传递的种类计算数据的分割阈值
def getMorse(data, kind_1, kind_0):
    list_1 = []
    list_0 = []
    type = data[0]
    count = 0
    list_total = []
    list_flag = []
    for elem in data:
        if (elem == type):
            count += 1
        else:
            if (type == 1):
                list_1.append(count)
                list_flag.append(1)
            else:
                list_0.append(count)
                list_flag.append(0)
            list_total.append(count)
            type = elem
            count = 1
    list_1.sort()
    list_0.sort()
    data_core_1 = k_tans(list_1, kind_1)
    data_core_0 = k_tans(list_0, kind_0)

    res = translate(list_total, list_flag, data_core_1, data_core_0)
    return res


def morseDecode(morse):
    # Morse init
    morseMap = {}
    morseMap['* -'] = 'a'
    morseMap['- * * *'] = 'b'
    morseMap['- * - *'] = 'c'
    morseMap['- * *'] = 'd'
    morseMap['*'] = 'e'
    morseMap['* * - *'] = 'f'
    morseMap['- - *'] = 'g'
    morseMap['* * * *'] = 'h'
    morseMap['* *'] = 'i'
    morseMap['*- - -'] = 'j'
    morseMap['- * -'] = 'k'
    morseMap['* - * *'] = 'l'
    morseMap['- -'] = 'm'
    morseMap['- *'] = 'n'
    morseMap['- - -'] = 'o'
    morseMap['* - - *'] = 'p'
    morseMap['- - * -'] = 'q'
    morseMap['* - *'] = 'r'
    morseMap['* * *'] = 's'
    morseMap['-'] = 't'
    morseMap['* * -'] = 'u'
    morseMap['* * * -'] = 'v'
    morseMap['* - -'] = 'w'
    morseMap['- * * -'] = 'x'
    morseMap['- * - -'] = 'y'
    morseMap['- - * *'] = 'z'
    morseMap['w'] = ' '
    alist = morse.split('a')
    res = ""
    for m in alist:
        res += morseMap[m]
    return res


# 主程序
filepath = "./"  # 添加路径
sampleRate = 20
filename = getFileName(filepath, '.wav')
for name in filename:
    wav = wave.open(filepath + name, 'rb')
    params = wav.getparams()
    nchannels, sampwidth, Fs, nframes = params[:4]
    strData = wav.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
    # 对信号进行压缩
    compressWaveData = getMeanByTime(waveData, sampleRate, params)

    # plot the wave
    time = np.arange(0, nframes) * (1.0 / Fs)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, waveData)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Single channel wavedata")
    plt.grid('on')  # 标尺，on：有，off:无。
    plt.subplot(2, 1, 2)
    time_c = np.arange(0, len(compressWaveData)) * (1.0 / Fs) * (Fs * sampleRate / 1000)
    plt.plot(time_c, compressWaveData)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Compress channel wavedata")
    decode, morse = getMorse(compressWaveData, 2, 3)
    plt.grid('on')  # 标尺，on：有，off:无。
    print(morse)
    str_res = morseDecode(decode)
    print(str_res)
    plt.show()
