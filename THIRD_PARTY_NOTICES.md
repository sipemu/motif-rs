# Third-Party Notices

motif-rs is licensed under the MIT License. See [LICENSE](LICENSE) for details.

This file documents the third-party dependencies used by motif-rs and the
academic works that inspired its algorithms.

---

## Dependencies

### realfft

- **License:** MIT
- **Authors:** Henrik Enquist
- **Repository:** <https://github.com/HEnquist/realfft>

Used for FFT-based sliding dot product computation (MASS algorithm).

### rayon

- **License:** MIT OR Apache-2.0
- **Authors:** Niko Matsakis, Josh Stone
- **Repository:** <https://github.com/rayon-rs/rayon>

Used for parallel computation of diagonal ranges.

### serde / serde_json

- **License:** MIT OR Apache-2.0
- **Authors:** Erick Tryzelaar, David Tolnay
- **Repository:** <https://github.com/serde-rs/serde>

Used for golden test data serialization (optional `validation` feature).

### criterion

- **License:** MIT OR Apache-2.0
- **Authors:** Jorge Aparicio, Brook Heisler
- **Repository:** <https://github.com/criterion-rs/criterion.rs>

Used for benchmarks (dev dependency only).

---

## Academic Attribution

motif-rs implements algorithms from the Matrix Profile research community.
The following papers describe the foundational techniques:

### Matrix Profile (STOMP)

> Yan Zhu, Zachary Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael Yeh,
> Gareth Funning, Abdullah Mueen, Philip Brisk, Eamonn Keogh.
> "Matrix Profile II: Exploiting a Novel Algorithm and GPUs to break the one
> Hundred Million Barrier for Time Series Motifs and Joins."
> *IEEE ICDM 2016.*

### SCRIMP / SCRIMP++

> Yan Zhu, Chin-Chia Michael Yeh, Zachary Zimmerman, Kaveh Kamgar,
> Eamonn Keogh.
> "Matrix Profile XI: SCRIMP++: Time Series Motif Discovery at Interactive
> Speeds."
> *IEEE ICDM 2018.*

### FLUSS / FLOSS

> Shaghayegh Gharghabi, Yifei Ding, Chin-Chia Michael Yeh, Kaveh Kamgar,
> Liudmila Ulanova, Eamonn Keogh.
> "Matrix Profile VIII: Domain Agnostic Online Semantic Segmentation at
> Superhuman Performance Levels."
> *IEEE ICDM 2017.*

### AAMP (Non-normalized Matrix Profile)

> Sara Abbaszade Masouleh, Eamonn Keogh.
> "Matrix Profile XXIV: Scaling Time Series Anomaly Detection to Trillions of
> Datapoints and Ultra-fast Arriving Data Streams."
> *ACM SIGKDD 2021.*

### MASS (Mueen's Algorithm for Similarity Search)

> Abdullah Mueen, Yan Zhu, Michael Yeh, Kaveh Kamgar, Krishnamurthy
> Viswanathan, Chetan Gupta, Eamonn Keogh.
> "The Fastest Similarity Search Algorithm for Time Series Subsequences under
> Euclidean Distance."
> *2017.*

### Snippets

> Shima Imani, Frank Madrid, Wei Ding, Scott Crouter, Eamonn Keogh.
> "Matrix Profile XIII: Time Series Snippets: A New Primitive for Time Series
> Data Mining."
> *IEEE BigKDD 2018.*

### MPdist

> Shaghayegh Gharghabi, Shima Imani, Anthony Bagnall, Amirali Darvishzadeh,
> Eamonn Keogh.
> "Matrix Profile XII: MPdist: A Novel Time Series Distance Measure to Allow
> Data Mining in More Challenging Scenarios."
> *IEEE ICDM 2018.*

### Chains (Time Series Chains)

> Yan Zhu, Makoto Imamura, Daniel Nikovski, Eamonn Keogh.
> "Matrix Profile VII: Time Series Chains: A New Primitive for Time Series
> Data Mining."
> *IEEE ICDM 2017.*

### STIMP (Pan Matrix Profile)

> Kaveh Kamgar, Shaghayegh Gharghabi, Eamonn Keogh.
> "Matrix Profile XV: Exploiting Time Series Consensus Motifs to Find
> Structure in Time Series Sets."
> *IEEE ICDM 2019.*

### Ostinato (Consensus Motif)

> Kaveh Kamgar, Shaghayegh Gharghabi, Eamonn Keogh.
> "Matrix Profile XV: Exploiting Time Series Consensus Motifs to Find
> Structure in Time Series Sets."
> *IEEE ICDM 2019.*

### MSTUMP (Multi-dimensional Matrix Profile)

> Chin-Chia Michael Yeh, Nickolas Kavantzas, Eamonn Keogh.
> "Matrix Profile VI: Meaningful Multidimensional Motif Discovery."
> *IEEE ICDM 2017.*

---

## stumpy

motif-rs validates its output against [stumpy](https://github.com/TDAmeritrade/stumpy),
a Python library for matrix profile computation.

> Sean M. Law.
> "STUMPY: A Powerful and Scalable Python Library for Time Series Data Mining."
> *Journal of Open Source Software, 4(39), 1504, 2019.*

- **License:** BSD 3-Clause
- **Repository:** <https://github.com/TDAmeritrade/stumpy>

stumpy is used as a reference implementation for testing only and is not
included in or linked by motif-rs.
