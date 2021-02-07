import os

for dir_ in ['examples', 'essentials', 'customizations', 'introductions']:
    os.system(f'for f in source/{dir_}/*.ipynb; do ./strip_kernel.sh $f; done')
