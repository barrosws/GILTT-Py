# QA-052 method references

The CI contract follows current Python packaging and CI guidance without binding the scientific package to one CI provider.

- Python Packaging User Guide, *Packaging Python Projects*: the standard packaging flow separates source configuration, build backends, distribution artifacts and installation. https://packaging.python.org/en/latest/tutorials/packaging-projects/
- Python Packaging User Guide, *Building and Publishing*: current PyPA build/publish guidance. https://packaging.python.org/en/latest/guides/section-build-and-publish/
- GitHub Docs, *Building and testing Python*: an example provider mapping in which Python versions are selected explicitly for CI jobs. https://docs.github.com/en/actions/tutorials/build-and-test-code/python

These references justify the engineering architecture only. They are not evidence that GILTT-Py has executed on an untested platform/Python matrix cell.
