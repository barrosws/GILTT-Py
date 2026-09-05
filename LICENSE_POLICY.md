# QA-054 license policy

No software license is selected by QA-054. The project reproducibility protocol requires an explicit software-license decision before public release. Until that decision is reviewed, `pyproject.toml` must not claim a software license and no `LICENSE` file may be fabricated from convention.

Software licensing, benchmark-data redistribution rights, and manuscript/supplement licensing are separate decisions. A license applicable to data does not automatically license the software, and a software license does not grant permission to redistribute third-party benchmark data.

When a software license is selected, its `pyproject.toml` representation must use the current SPDX license-expression mechanism and the distributed license text must be included according to the packaging metadata contract.
