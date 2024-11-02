import sys
import pickle


def build_cooc(source_folder, destination_file):
    wordlist = {}
    for line in open('en_words.txt').readlines():
        wordlist[line.strip()] = 1
    file_list = open(f'{source_folder}/file_list.txt').readlines()
    f = open(destination_file, 'w')
    for file in file_list:
        file = file.strip()
        print(file)
        try:
            w2w = pickle.load(open(f'{source_folder}/{file}', 'rb'))
            src_dict = {v: k for k, v in w2w[0].items()}
            dst_dict = {k: v for k, v in w2w[1].items()}
            for en_index in src_dict:
                for tl_index in w2w[2][en_index]:
                    en_word = src_dict[en_index]
                    if en_word not in wordlist: continue
                    tl_word = dst_dict[tl_index]
                    if len(en_word) > 1 and len(tl_word) > 1 and len(en_word) < 20 and len(tl_word) < 20:
                        f.write(f"{tl_word}\t{en_word}\t10\n")
        except Exception as e:
            print(e)

    f.close()


if __name__ == '__main__':
    if len(sys.argv) == 3:
        build_cooc(sys.argv[1], sys.argv[2])
