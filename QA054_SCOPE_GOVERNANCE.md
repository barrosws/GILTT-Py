# QA-054 Scope Governance

QA-054 governs scientific-software metadata, citation, authorship-review state, and licensing state. It does not modify or revalidate scientific equations, physics, numerical methods, benchmark outputs, sensitivity designs, or uncertainty analyses.

The authorship of a dissertation, paper, legacy code, or source document does not by itself establish authorship of GILTT-Py software. Software authorship and ORCIDs require explicit review. No ORCID may be guessed.

`CITATION.cff.template` is a non-release template only. The target format is Citation File Format 1.2.0. A public `CITATION.cff` may be created only after unresolved authorship/release fields are reviewed and schema validation is performed. Placeholder values and fabricated DOI values are prohibited in public release metadata.

Software license selection remains HOLD. Data redistribution rights are audited separately. A data license is not a software license by default. Current PyPA/PEP 639 activation uses an SPDX license expression plus `license-files`; for the current setuptools backend that syntax requires setuptools 77.0.3 or later. Because no software license has yet been selected, QA-054 does not alter the current build requirement or activate license metadata; that backend upgrade is a release-engineering HOLD.

The development package remains `0.0.0.dev1`; public release version, release date, DOI and repository URL are deferred to the release gate.

NO TARGET TUNING. Engineering metadata QA cannot expand the scientific claim envelope.
