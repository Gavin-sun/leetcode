import numpy as np

def loaddataset(Filename):
    fp = open(Filename,encoding='utf-8')
    a = fp.readlines()
    print(a)
    fp.close()

    dataset = []
    for i in a[1:]:
        b = i.strip().split()
        dataset.append(b[1:])
    #返回数据和属性标签
    return dataset, a[0].strip().split()[1:]

def statistics(dataset, n):
    #记录统计结果
    count = {}
    for i in dataset:
        if i[n] not in count:
            #列表第一个位置为正例个数，第二个位置是反例个数
            if i[-1] == '是':
                count[i[n]] = [1, 0]
            else:
                count[i[n]] = [0, 1]
        else:
            if i[-1] == '是':
                count[i[n]][0] += 1
            else:
                count[i[n]][1] += 1
    #计算各个属性取值为正例的概率和反例的概率
    for i in count:
        n = count[i][0] + count[i][1]
        count[i][0] = count[i][0] / n
        count[i][1] = count[i][1] / n
    return count


if __name__ == '__main__':
    dataset, labelset = loaddataset(r'..\resource\xigua.txt')
    count_sum = []
    for i in range(len(dataset[0])-1):
        count_sum.append(statistics(dataset, i))
    a = list(np.array(dataset)[:,-1])
    y = a.count('是')
    n = a.count('否')

    #记录正确预测的个数
    rightcount = 0
    for i in range(len(dataset)):
        #记录分为正类的概率
        py = 1
        #记录分为反类的概率
        pn = 1
        #p(y|xx,xx,xx,xx)=p(xx|y)(pxx|y)(pxx|y)(pxx|y)*p(y)
        for j in range(len(dataset[i])-1):
            b = count_sum[j]
            py *= b[dataset[i][j]][0]
            pn *= b[dataset[i][j]][1]
        py *= y / (y + n)
        pn *= n / (y + n)
        if py >= pn:
            flag = '是'
        else:
            flag = '否'
        if flag == a[i]:
            rightcount += 1
        print('%s  %s'%(flag, a[i]))
    print("正确率为%s"%(rightcount / len(dataset)))
