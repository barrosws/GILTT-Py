# QA-053 Scope Governance

QA-053 governs dependency-resolution and lock semantics only. It does not modify or revalidate scientific equations, physical parameterizations, benchmark observations, sensitivity distributions or model-form choices.

A local installed-version snapshot is **LOCAL_RESOLUTION** evidence. It is not a hermetic lock because it does not identify immutable artifact hashes, and it is not cross-platform evidence. A `pip freeze` output is not a universal lock and must never be promoted as one.

A future **HERMETIC_HASH_LOCK** requires exact versions plus artifact SHA-256 hashes and successful installation/replay in the target environment. Cross-platform targets require separate compatible artifact sets or another independently audited lock mechanism.

NO TARGET TUNING remains binding.
