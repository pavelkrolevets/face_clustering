"""
Downloads the LFW face dataset and extracts it to ./data/lfw/.

If the primary download URL is not accessible, the script automatically
falls back to mirrors (see MIRRORS below).
"""

import os
import shutil
import tarfile
import tempfile
import urllib.request

PRIMARY_URL = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
MIRRORS = [
    # Mirrors to fall back to when the primary URL is not accessible.
    # The original UMass host (vis-www.cs.umass.edu) has been decommissioned,
    # so the default fallback is the full LFW release (13233 images) mirrored
    # on Hugging Face:
    'https://huggingface.co/datasets/DerrickUnleashed/LFW/resolve/main/lfw.tgz',
]

DATA_DIR = './data/lfw'


def download_file(url, destination):
    """Download a file from url and save it to destination."""
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with open(destination, 'wb') as f:
        f.write(data)
    return destination


def get_urls():
    """Return the primary URL followed by any configured mirrors."""
    return [PRIMARY_URL] + MIRRORS


def download_and_extract():
    """Download the LFW dataset and extract it into ./data/lfw/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tgz_path = os.path.join(tmp_dir, 'lfw.tgz')
        downloaded = False
        errors = []
        for url in get_urls():
            try:
                download_file(url, tgz_path)
                downloaded = True
                print('Downloaded LFW dataset from', url)
                break
            except Exception as e:
                errors.append(f'{url}: {e}')
        if not downloaded:
            raise RuntimeError(
                'Failed to download the LFW dataset from any source: ' + '; '.join(errors))

        with tarfile.open(tgz_path, 'r:gz') as tar:
            members = [m for m in tar.getmembers() if m.name]
            if not members:
                raise RuntimeError('The LFW dataset archive is empty.')
            try:
                tar.extractall(tmp_dir, filter='fully_trusted')
            except TypeError:
                # Python < 3.12 has no filter argument.
                tar.extractall(tmp_dir)
            top_level = members[0].name.split('/')[0]
            src_dir = os.path.join(tmp_dir, top_level)
            if not os.path.isdir(src_dir):
                # Archive is flat, files live directly in the temporary directory.
                src_dir = tmp_dir
            for entry in sorted(os.listdir(src_dir)):
                source = os.path.join(src_dir, entry)
                destination = os.path.join(DATA_DIR, entry)
                if os.path.exists(destination):
                    # Already present from a previous run; leave it as is.
                    continue
                shutil.move(source, destination)

    print(f'LFW dataset extracted to {DATA_DIR}')


if __name__ == '__main__':
    download_and_extract()
