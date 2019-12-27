DT42 dyda library and application

[![pipeline status](https://gitlab.com/DT42/galaxy42/dt42-dyda/badges/develop/pipeline.svg)](https://gitlab.com/DT42/galaxy42/dt42-dyda/commits/develop)

# Build Wheel Package

Execute the command in project root dir:

```
$ python3 setup.py bdist_wheel
```

The wheel package will be created in dist/.

# Build Debian Package

Execute the command in project root dir:

```
$ git checkout debian
$ debuild -us -uc
```

The Debian package will be created in ../.
