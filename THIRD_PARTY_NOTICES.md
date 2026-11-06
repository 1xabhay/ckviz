# Third-party notices

ckviz itself is under the MIT License (see [LICENSE](LICENSE)). This file covers material in this repository that
comes from others, software it relies on but does not distribute, and the papers it reproduces.

## Absolutist lexicon

The word list in `examples/absolutist-*.yaml` is adapted from Table S2 of Al-Mosaiwi, M., & Johnstone, T. (2018),
In an Absolute State, *Clinical Psychological Science* 6(4), 529-542, <https://doi.org/10.1177/2167702617747074>,
CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>).

Changes: four entries converted to wildcards (`absolut*`, `entire*`, `total*`, `everyone*`). The 63-phrase exclusion
list was reconstructed by ckviz from the authors' data, Figshare doi:10.6084/m9.figshare.4743547 and
10.6084/m9.figshare.4743715, CC BY 4.0. Not LIWC content.

## ConvoKit

Pipeline settings and expected test values reproduce ConvoKit example notebooks.

```
Copyright (C) 2017-2024 The ConvoKit Developers.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Not distributed here

These are installed or fetched separately, under their own terms:

- **DLATK** (GPL-3.0): run as a separate process via `uvx` by the `ckviz-dlatk` plugin.
- **Unidecode** (GPL-2.0-or-later): required by ConvoKit. Don't publish a Docker image or frozen bundle of ckviz
  without meeting the GPL's terms.
- **psycopg** (LGPL-3.0-only).
- **ConvoKit's bundled LIWC-derived marker patterns and Liu opinion lexicon**: terms set by their owners; not vendored
  here.

## Papers reproduced

Write-ups are in [`examples/papers/`](examples/papers/).

- Al-Mosaiwi, M., & Johnstone, T. (2018). In an Absolute State. *Clinical Psychological Science* 6(4), 529-542.
- Danescu-Niculescu-Mizil, C., Lee, L., Pang, B., & Kleinberg, J. (2012). Echoes of power. WWW 2012.
  <https://arxiv.org/abs/1112.3670>
- Danescu-Niculescu-Mizil, C., Sudhof, M., Jurafsky, D., Leskovec, J., & Potts, C. (2013). A computational approach to
  politeness. ACL 2013. <https://arxiv.org/abs/1306.6078>
- Zhang, J., Chang, J. P., Danescu-Niculescu-Mizil, C., Dixon, L., Hua, Y., Taraborelli, D., & Thain, N. (2018).
  Conversations Gone Awry. ACL 2018. <https://arxiv.org/abs/1805.05345>
