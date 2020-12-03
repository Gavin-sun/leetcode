import re
import pandas as pd
import xlwt
import pickle
ws = pickle.load(open("E:\git_code\Machine_learning\py\hatespeech\data\ws.pkl","rb"))
def clean(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"!|RT|#|&128514;", "", text,flags=re.S)
    text = re.sub(r"&amp;", "", text,flags=re.S)
    text = re.sub(r"(&)?(#)\d*(;| |$)", "", text,flags=re.S) # 去除表情符号
    text = re.sub(r"(&)\d*(;)", "", text,flags=re.S) # 去除表情符号
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)       # 去除网址
    text = re.sub(r"\s+", " ", text) # 合并正文中过多的空格
    return text.strip().lower()

train_df = pd.read_excel(r'E:\git_code\Machine_learning\py\hatespeech\data\1.xls', sep='\t', nrows=24783,encoding='utf-8')

# f = xlwt.Workbook()
# sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
# rowTitle = [u'class',u'tweet']
#
# for i in range(0,len(rowTitle)):
#     sheet1.write(0,i,rowTitle[i])
#
#
# for k in range(1,len(train_df['tweet'])):    #先遍历外层的集合，即每行数据
#     sheet1.write(k,0,int(train_df['class'][k]))
#     sheet1.write(k,1,clean(train_df['tweet'][k]))
#
# #保存文件的路径及命名
# f.save(r'C:\Users\Gavin\Desktop\2.xls')


f = xlwt.Workbook()
sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
rowTitle = [u'class',u'tweet']

for i in range(0,len(rowTitle)):
    sheet1.write(0,i,rowTitle[i])


for k in range(1,len(train_df['tweet'])):    #先遍历外层的集合，即每行数据
    sheet1.write(k,0,int(train_df['class'][k]))

    content = [str(train_df["tweet"][k]).split(" ")]
    sheet1.write(k,1,"".join('%s' %ws.transform(i,max_len=20) for i in content))

#保存文件的路径及命名
f.save(r'C:\Users\Gavin\Desktop\2.xls')
