"""Download and extract NINAPRO databases 1 to 3."""
import os
from multiprocessing import Pool
import time
import re
import shutil
from itertools import product
import zipfile
import pathlib
import requests

MAIN_URL = "https://datadryad.org/resource/doi:10.5061/dryad.1k84r"
HOST = "https://datadryad.org"
CWD = os.getcwd()
PATHS = [CWD + "/databases/db1",
         CWD + "/databases/db2",
         CWD + "/databases/db3"]


def get_filelinks(url=MAIN_URL):
    """Get links for downloading subjects zip files.

    Parameters:
        url (:obj:`str`, optional, *default* =MAIN_URL): url for getting
            the files links

    """
    main_page = requests.get(url).text
    all_links = re.findall("(/bitstream/handle/.+?\.zip)", main_page)
    db1_links = [HOST+filelink for filelink in all_links if 'DB1' in filelink]
    db2_links = [HOST+filelink for filelink in all_links if 'DB2' in filelink]
    db3_links = [HOST+filelink for filelink in all_links if 'DB3' in filelink]
    return [db1_links, db2_links, db3_links]


def get_file(url, path):
    """Download subjects zip files.

    Parameters:
        url (:obj:`str`): url for getting the files links
        path (:obj:`str`): path to the folder where to download and extract the files

    """
    start_time = time.time()
    local_filename = url.split("/")[-1]
    print("Starting download of file {}...".format(local_filename))
    response = requests.get(url, stream=True)
    with open(path+"/"+local_filename, "wb") as mfile:
        shutil.copyfileobj(response.raw, mfile)
    print(local_filename + " took " + str(time.time() - start_time) + " seconds")
    extract_zips(path+"/"+local_filename, path)


def extract_zips(zipfilename, path):
    """Extract a dataset zipped file.

    Parameters:
        zipfilename (:obj:`str`): path of the file to extract
        path (:obj:`str`): path to the folder where to extract the files

    """
    zpfile = zipfile.ZipFile(zipfilename, "r")
    files_list = []
    for zfile in zpfile.namelist():
        if "MACOSX" not in zfile:
            if zfile.endswith(".mat") or zfile.endswith(".MAT"):
                files_list.append(zpfile.extract(zfile, path))
    for mats in files_list:
        parts = mats.split("/")
        to_rmpath = "/".join(parts[:-1])
        shutil.move(mats, "{}/{}".format(path, parts[-1]))
    if to_rmpath != path:
        try:
            shutil.rmtree(to_rmpath)
        except Exception as error:
            print ("Error while removing path {}: {}".format(to_rmpath, error))
    try:
        os.remove(zipfilename)
    except Exception as error:
        print("Error while removing zip file {}: {}".format(zipfilename, error))


def download_dbs(paths=PATHS):
    """Download and extract all subjects zipped files.

    Parameters:
        url (:obj:`str`): url for getting the files links
        paths (:obj:`str`, optional, *default* =PATHS): list of paths to the folders
            each database datasets will be downloaded and extracted

    """
    for path_dir in paths:
        pathlib.Path(path_dir).mkdir(parents=True, exist_ok=True)
    for count, database_urls in enumerate(get_filelinks()):
        print("Downloading files from database {}...".format(count + 1))
        path = PATHS[count]
        start_time = time.time()
        with Pool(processes=10) as pool:
            pool.starmap(get_file, product(database_urls, [path]))
        print("\nDownloading entire database {} took {} seconds.".
              format(count + 1, + time.time() - start_time))


if __name__ == "__main__":
    download_dbs()
