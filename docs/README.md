If new module files have been created, execute the following command before building the documentation:

```bash
sphinx-apidoc -f -o source/api ..
```

To build the docs, run:

```bash
make html
```