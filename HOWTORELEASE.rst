Release Procedure
-----------------

#. Update CHANGELOG.rst to reflect release date. Every pull request should be documented under one of these categories.

    - breaking changes
    - deprecations
    - new features
    - enhancements
    - documentation
    - internal changes
    - bug fixes

#. Tag a release and push to github::

    $ git tag -a v0.0.x -m "Version 0.0.x"
    $ git push origin master --tags

#. Build and publish release on PyPI::

    $ git clean -xfd  # remove any files not checked into git
    $ python setup.py sdist bdist_wheel --universal  # build package
    $ twine upload dist/*  # register and push to pypi

#. Update the stable branch (used by ReadTheDocs)::

    $ git checkout stable
    $ git rebase master
    $ git push -f origin stable
    $ git checkout master
