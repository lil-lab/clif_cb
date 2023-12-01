import os

heu_err_cats = dict()
raw_err_cats = dict()
for filename in os.listdir('analysis/'):
    if filename.startswith('annotated_all_'):
        datename = '_'.join(filename.split('.')[0].split('_')[1:])

        with open('analysis/' + filename) as infile:
            text = infile.read()
        """
        rg = text.count("(heu, -): rb, good")
        rb = text.count("(heu, -): rb, bad")
        g = text.count("(heu, -): good")
        b = text.count("(heu, -): bad")

        print(f'{datename}\t{rg}\t{rb}\t{g}\t{b}')
        """
        """
        for line in text.split('\n'):
            if "): " in line:
                ann = line.split('):')[1].strip()
                if ann:
                    dict_to_use = heu_err_cats if 'heu' in line else raw_err_cats
                    if ann not in dict_to_use:
                        dict_to_use[ann] = 0
                    dict_to_use[ann] += 1
                    
        """
        # """
        rpg = text.count("(raw, +): good")
        hpg = text.count("(heu, +): good")
        rpb = text.count("(raw, +): bad")
        hpb = text.count("(heu, +): bad")
        rng = text.count("(raw, -): good")
        hng = text.count("(heu, -): good")
        rnb = text.count("(raw, -): bad")
        hnb = text.count("(heu, -): bad")

        print(
            f'{datename}\t{rpg}\t{hpg}\t{rpb}\t{hpb}\t{rng}\t{hng}\t{rnb}\t{hnb}'
        )
        # """
        """
        current_exid = ''
        with open(f'analysis/annotated_negative_errors_{datename}.txt',
                  'w') as ofile:
            for line in text.split('\n'):
                if line.startswith('-----'):
                    current_exid = line.split(' ')[2]

                if ': bad' in line:
                    if current_exid:
                        ofile.write(current_exid + '\n')
                        current_exid = ''
                    else:
                        print(current_exid)
                        
        """

print('error types for heuristic feedback')
for err, count in heu_err_cats.items():
    print(f'{err}\t{count}')
print('')
print('error types for raw feedback')
for err, count in raw_err_cats.items():
    print(f'{err}\t{count}')
