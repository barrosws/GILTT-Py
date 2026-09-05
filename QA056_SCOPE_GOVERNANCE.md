# QA-056 scope governance — documentation and provenance completeness audit

QA-056 is an audit gate, not an archival-release readiness gate. It is derived from the frozen project master-plan requirements for the reproducible computational pipeline (WP16) and Zenodo-ready archival release (WP18).

The audit must classify every required documentation/provenance role as either PRESENT or HOLD. Missing artifacts must never be silently treated as complete.

QA-056 does not select software authors, ORCIDs, license, DOI, release date, public semantic version or data redistribution rights. Those fields remain under the QA-054 governance HOLDs. It does not alter scientific equations, models, numerical tolerances, sensitivity/UQ specifications, claim-envelope semantics or the historical/modern branch separation.

The gate may PASS when the *audit itself* is complete and internally consistent even though archival-release readiness remains false because explicit HOLDs remain.
