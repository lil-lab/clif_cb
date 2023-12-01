import os

GR_PREFIX = 'game_recordings/'

gid_to_did = dict()

for data_dir in os.listdir(GR_PREFIX):
    if data_dir.endswith('.zip') or data_dir.endswith(
            '.txt') or data_dir.startswith('.'):
        continue
    for game_file in os.listdir(os.path.join(GR_PREFIX, data_dir, 'games')):
        if game_file.endswith('.pkl'):
            game_id = game_file.split('.')[0]

            gid_to_did[game_id] = data_dir

exid_to_attr = dict()
all_attrs = set()

with open('analysis/annotated_language.txt') as infile:
    ex_id: str = ''
    current_attrs = list()

    for line in infile.readlines():
        if line.startswith('-----'):
            if current_attrs:
                exid_to_attr[ex_id] = sorted(current_attrs)
                current_attrs = list()

            ex_id = line.split(' ')[1]
        if line.startswith('. '):
            attr = line[2:].strip()
            current_attrs.append(attr)

            all_attrs.add(attr)

            if gid_to_did[ex_id.split('-')
                          [0]] == '11_15' and "incorrect card" in attr:
                print(ex_id)

    if current_attrs:
        exid_to_attr[ex_id] = sorted(current_attrs)
        current_attrs = list()

did_attrs = dict()
attr_patterns = dict()
all_attr_patterns = set()
for exid, attrs in exid_to_attr.items():
    gid = exid.split('-')[0]
    did = gid_to_did[gid]

    if did not in did_attrs:
        did_attrs[did] = {attr: 0 for attr in all_attrs}
    if did not in attr_patterns:
        attr_patterns[did] = dict()

    pattern = len(attrs)
    if pattern not in attr_patterns[did]:
        attr_patterns[did][pattern] = 0
    attr_patterns[did][pattern] += 1
    all_attr_patterns.add(pattern)

    for attr in attrs:
        did_attrs[did][attr] += 1

for did, attr_dict in did_attrs.items():
    print(f'\t{did}')
    for attr, count in sorted(attr_dict.items(), key=lambda x: x[0]):
        print(f'{count}\t{attr}')
    print('')

#    print(f'\t{did}')
#    for attr_pattern in all_attr_patterns:
#        if attr_pattern in attr_patterns[did]:
#            num = attr_patterns[did][attr_pattern]
#        else:
#            num = 0
#        print(f'{num}\t{attr_pattern}')
