# Security policy

uccap is a computation library. It reads the files and data you give it, runs locally, and never
sends your data anywhere.

## Supported versions

Security fixes are made for the latest release.

## Reporting a vulnerability

Please report a vulnerability privately through GitHub: on the repository's **Security** tab, choose
**Report a vulnerability**. Do not open a public issue for it. You will get a reply on the report,
and the fix and an advisory will be published once it is resolved.

Examples of what to report: a crafted Excel or CSV file that makes `read_table` execute code or read
other files, or a way for the package to send data off the computer.
