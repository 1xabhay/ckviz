# Security

## Reporting

Report vulnerabilities through a [private advisory on GitHub](https://github.com/1xabhay/ckviz/security/advisories/new),
not a public issue. Don't include participant data in a report.

## Threat model

ckviz is a single-user app meant to run on your own machine.

- It binds to localhost by default.
- The `Host` check stops DNS rebinding. It isn't access control.
- Mutating API calls need an `X-CKVIZ: 1` header, as a guard against cross-site requests.
- There is no login. Anyone who can reach the port can use the app, so don't expose it to a network.
- `dir` and `convokit` sources are trusted input, because ConvoKit can unpickle files in a corpus. Only load corpora
  from sources you trust.
- `data/` holds participant text: downloads, snapshots and pickled caches. Keep it writable only by you, and never
  commit participant data.
