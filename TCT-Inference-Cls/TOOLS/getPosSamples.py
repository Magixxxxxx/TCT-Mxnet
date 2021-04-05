import os
import sys

def isPos(root,f):
    labellist = ['1','2','3','4','5','6','8','9','15']
    try:
        with open(root+'/'+f,'r',) as ann:
            target_num = ann.readline().strip('\n')
            if target_num.isdigit():
                target_num = int(target_num)
            else:
                return -1
            
            for i in range(target_num):
                tp = ann.readline()[0]
                if any(label == tp for label in labellist):
                    return 1
            return 0
    except UnicodeDecodeError as e:
        print(e)
        print(root, f)
        return -1

def getPosSamples(target_dir, output_file0, output_file1):
    flist = os.walk(target_dir)

    total_num = 0
    pos_num = 0
    neg_num = 0
    o_f0 = open(output_file0,'w',encoding='utf-8')
    o_f1 = open(output_file1,'w',encoding='utf-8')

    for root,dirs,files in flist:
        for f in files:
            if f.endswith('.txt'):
                if not os.path.exists(root + '/' + f[:-4]+'.jpg'):
                    print(root + '/' + f[:-4]+'.jpg','not exist')
                    continue
                p = isPos(root,f)
                total_num += 1

                if p==1:
                    pos_num += 1 
                    o_f1.write('{}/{}.jpg\n'.format(root, f[:-4]))
                elif p==0:
                    neg_num += 1
                    o_f0.write('{}/{}.jpg\n'.format(root, f[:-4]))

                sys.stdout.write("\r {}".format(total_num))
                sys.stdout.flush()
                
    o_f0.close()
    o_f1.close()

    print('{} {} {} {}'.format(pos_num,neg_num,total_num,total_num-pos_num-neg_num))

if __name__ == '__main__':
    target_dir = sys.argv[1]
    output_file0 = sys.argv[2] if len(sys.argv)>2 else 'negSamples.txt'
    output_file1 = sys.argv[3] if len(sys.argv)>3 else 'posSamples.txt'
    getPosSamples(target_dir,output_file0,output_file1)
