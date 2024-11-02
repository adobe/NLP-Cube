import sys

import requests


def _download_file(url, destination):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192 * 1024):
                f.write(chunk)
    f.close()
    return


def download_targets(src_lang, output_path):
    lines = open('all_langs.txt').readlines()
    filenames = []
    for line in lines:
        [src, dst] = line.strip().split('-')
        if src == src_lang:
            filenames.append(f'{src}-{dst}')
    index = 46
    total = len(filenames)
    for filename in filenames[46:]:
        index += 1
        print(f"Downloading {filename} ({index}/{total})")
        url = f"https://mk.kakaocdn.net/dn/kakaobrain/word2word/{filename}.pkl"
        destination = f"{output_path}/{filename}.pkl"
        _download_file(url, destination)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage download_w2w.py <folder>")
    else:
        output_path = sys.argv[1]
        download_targets('en', output_path)
