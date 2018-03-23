#
# Author: Ruxandra Burtica
#
# Copyright (c) 2018 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import shutil
import tarfile

import requests


def download_file(url, file_name):
    # First remove the archive_name file if it exists.
    if os.path.exists(file_name):
        os.remove(file_name)
    request = requests.get(url)
    with open(file_name, "wb") as fd:
        fd.write(request.content)


def download_and_extract_archive(url, file_location):
    archive_name = file_location + '.tgz'

    # Download file
    download_file(url, archive_name)

    # Extract from archive and remove archive.
    tar = tarfile.open(archive_name)
    tar.extractall(path=file_location)
    tar.close()
    os.remove(archive_name)

    # Move content to the pre-defined location.
    directories = [
        name for name in os.listdir(file_location)
        if os.path.isdir(os.path.join(file_location, name))]
    sub_directory = directories[0]
    for filename in os.listdir(os.path.join(file_location, sub_directory)):
        shutil.move(
            os.path.join(file_location, sub_directory, filename),
            os.path.join(file_location, filename))
