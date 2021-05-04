import os
name = "mh45.dat"
#Making the list

def get(name):

    if os.path.exists(str(name)):
        # Reading a file
        with open(str(name),'r') as g:# Open file for reading
            pts = []
            for i in g.readlines():
                line = i.strip('\n').split(' ')
                ln = []
                for j in range(len(line)):
                    line[j] = line[j].lstrip()
                    if len(line[j]) != 0:
                        ln.append(line[j])
                    else:
                        continue
                #print(ln)
                pts.append(ln)
        #print(pts)
        #getting dat header
        header = ' '.join(pts[0])
        del pts[0]

    else:
        exit()

    return(header,pts)


print(get(name)[0])
print(get(name)[1])
