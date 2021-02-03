import os

os.system('for f in **/**/*.ipynb; do ./strip_kernel.sh $f; done')
