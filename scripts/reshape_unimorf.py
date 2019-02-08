import sys


def get_upos(unimorf):
    upos_list = ['N', 'PROPN', 'ADJ', 'PRO', 'CLF', 'ART', 'DET', 'V', 'ADV', 'AUX', 'V.PTCP', 'V.MSDR', 'V.CVB', 'ADP',
                 'COMP', 'CONJ', 'NUM', 'PART', 'INTJ']
    parts = unimorf.split(';')
    for part in parts:
        if part in upos_list:
            return part
    # print(unimorf)
    return "_"


def get_attrs(unimorf, upos):
    parts = unimorf.split(';')
    if upos not in parts:
        print (upos)
    else:
        parts.remove(upos)
    parts = sorted(parts)
    attrs = ''
    for part in parts:
        attrs += part + ';'
    if attrs == '':
        attrs = '_;'
    attrs = attrs[:-1]
    return attrs


def update_sigmorphon(input_file, output_file):
    inpf = open(input_file, 'r')
    outf = open(output_file, 'w')
    for line in inpf:
        parts = line.split('\t')
        if len(parts) < 3:
            outf.write(line)
        else:
            unimorf_label = parts[5]
            upos = get_upos(unimorf_label)
            attrs = get_attrs(unimorf_label, upos)

            o_line = ''
            for part, index in zip(parts, range(len(parts))):
                if index == 3:
                    o_line += upos + '\t'
                elif index == 5:
                    o_line += attrs + '\t'
                else:
                    o_line += part + '\t'

            o_line = o_line[:-1]
            outf.write(o_line)

    inpf.close()
    outf.close()


def display_help():
    sys.stdout.write('Usage: reshape_unimorf.py <input file> <output_file>\n')


if len(sys.argv) != 3:
    display_help()
else:
    update_sigmorphon(sys.argv[1], sys.argv[2])
