'''Download and extract NINAPRO datasets 1 to 3.'''
import os
from multiprocessing import Pool
import time
import re
import shutil
from itertools import product
import zipfile
import pathlib
import requests

MAIN_URL = 'https://datadryad.org/resource/doi:10.5061/dryad.1k84r'
HOST = 'https://datadryad.org'
PATHS = ['datasets/db1', 'datasets/db2', 'datasets/db3']


def get_filelinks(url=MAIN_URL):
    '''Get links for downloading subjects zip files'''
    main_page = requests.get(url).text
    all_links = re.findall("(/bitstream/handle/.+?\.zip)", main_page)
    db1_links = [HOST+filelink for filelink in all_links if 'DB1' in filelink]
    db2_links = [HOST+filelink for filelink in all_links if 'DB2' in filelink]
    db3_links = [HOST+filelink for filelink in all_links if 'DB3' in filelink]
    return [db1_links, db2_links, db3_links]


def get_file(url, path):
    start_time = time.time()
    local_filename = url.split('/')[-1]
    print('Starting download of file {}...'.format(local_filename))
    r = requests.get(url, stream=True)
    with open(path+'/'+local_filename, 'wb') as mfile:
        shutil.copyfileobj(r.raw, mfile)
    print(local_filename + " took " + str(time.time() - start_time) + " seconds")
    extract_zips(path+'/'+local_filename, path)


def extract_zips(zipfilename, path):
    zf = zipfile.ZipFile(zipfilename, 'r')
    files_list = []
    for zfile in zf.namelist():
        if 'MACOSX' not in zfile:
            if zfile.endswith('.mat') or zfile.endswith('.MAT'):
                files_list.append(zf.extract(zfile, path))
    for mats in files_list:
        parts = mats.split('/')
        to_rmpath = '/'.join(parts[:-1])
        shutil.move(mats, '{}/{}'.format(path, parts[-1]))
    if to_rmpath != path:
        try:
            shutil.rmtree(to_rmpath)
        except Exception as error:
            print ('Error while removing path {}: {}'.format(to_rmpath, error))
    try:
        os.remove(zipfilename)
    except Exception as error:
        print('Error while removing zip file {}: {}'.format(zipfilename, error))


def download_dbs(paths=PATHS):
    for path_dir in paths:
        pathlib.Path(path_dir).mkdir(parents=True, exist_ok=True)
    for count, database_urls in enumerate(get_filelinks()):
        print("Downloading files from database {}...".format(count + 1))
        path = PATHS[count]
        start_time = time.time()
        with Pool(processes=10) as pool:
            pool.starmap(get_file, product(database_urls, [path]))
        print("\nDatabase {} took {} seconds.".
              format(count + 1, + time.time() - start_time))


if __name__ == "__main__":
    download_dbs()
