f = open('filelist.dat', 'w')

times = ['0300', '0900', '1500', '2100']

f.write(f'https://lasco-www.nrl.navy.mil/content/retrieve/polarize/2009_03/vig/c2/C2-PB-20090315_2104.fts\n')

for d in range(16, 30):
    for t in times:
        if (d != 30 and t != '2100'):
            f.write(f'https://lasco-www.nrl.navy.mil/content/retrieve/polarize/2009_03/vig/c2/C2-PB-200903{d}_{t}.fts\n')

f.close()