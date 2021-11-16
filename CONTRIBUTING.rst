Contributing Procedure
----------------------

1. Browse issues at https://github.com/ahuang11/ahlive/issues, or submit a new one.
2. Select the one that interests you.
3. (Optional) Install Python executable if you don't already have; example for Linux.::

    $ wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    $ bash Miniforge3-Linux-x86_64.sh -b
    $ source ~/.bashrc

4. (Optional) Create a virtual environment for ahlive.::

    $ conda create python=3.9 -n ahlive_dev
    $ conda activate ahlive_dev

5. Clone ahlive locally.::

    $ git clone https://github.com/ahuang11/ahlive.git

6. Install ahlive and its dependencies.::

    $ cd ahlive
    $ pip install -e .
    $ conda env update ci/environment_ci.yml
    $ pre-commit install

7. Make changes to the code for fixing the issue.
8. Commit your changes and push.::

    $ git checkout -b "name_of_branch"
    $ git add .
    $ git commit -m "your message here"
    $ git push origin "name_of_branch"

9. In the pull request text, please provide a link to the GitHub issue.
10. Wait for tests to pass and a code review if needed.
11. Merged! Thank you for your contribution!
