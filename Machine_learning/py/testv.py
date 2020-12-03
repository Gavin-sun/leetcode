if __name__ == '__main__':
    a = {1:2,3:4,7:1}
    print(sorted(a.items(),key=lambda x:x[-1],reverse=True)[:2])
