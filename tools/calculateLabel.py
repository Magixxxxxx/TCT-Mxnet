import os
from collections import defaultdict

cat_counter = defaultdict(int)

for root,dirs,files in os.walk("/root/commonfile/TCTAnnotatedData/TCTAnnotated20210331"):
    for f_name in files:
        if '.txt' in f_name:
            f_path = os.path.join(root,f_name)
            with open(f_path) as f:
                n = f.readline()
                for label in f:
                    cat,_,_,_,_ = label.split()
                    cat_counter[int(cat)] += 1

for cat,num in sorted(cat_counter.items()):
    print(cat, num)

        

