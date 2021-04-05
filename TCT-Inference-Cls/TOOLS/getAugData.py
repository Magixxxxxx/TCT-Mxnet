import sys
from random import shuffle

neg_infer_file = sys.argv[1]
pos_paths_file = sys.argv[2]
r = float(sys.argv[3]) # 阈值

gt = 0.0 # 1/0

p_to_n = 0.5
train_to_test = 10

nlist = []
plist = []

with open(neg_infer_file, encoding = 'utf-8') as n_f:
    with open(pos_paths_file, encoding = 'utf-8') as p_f:
        #get hard samples
        for row in n_f:
            path,_,p = row.strip('\n').rpartition(' ')
            if abs(float(p)-gt)>0.4:
                nlist.append('{} {}'.format(path, int(gt)))

        #get pos samples
        nlen = len(nlist)
        plen = round(nlen * p_to_n)
        plist = p_f.readlines() 
        shuffle(plist)   
        plist = plist[:plen]
        for i in range(len(plist)):
            plist[i] = plist[i].strip('\n') + ' 1'

        #mix up
        all_list = plist + nlist
        shuffle(all_list)
        print(all_list[:50])
        div = round( len(all_list) / train_to_test)
        with open('all.txt','w', encoding = 'utf-8') as o_f:
            o_f.writelines([line+'\n' for line in all_list])
        with open('train.txt','w', encoding = 'utf-8') as o_f:
            o_f.writelines([line+'\n' for line in all_list[div:]])
        with open('test.txt','w', encoding = 'utf-8') as o_f:
            o_f.writelines([line+'\n' for line in all_list[:div]])  
