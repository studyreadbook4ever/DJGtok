# Third-party notices

DJGtok's original source code and documentation are offered under the [MIT license](LICENSE). This does not relicense external datasets, dependency packages, papers, trademarks, or material that a user downloads separately.

## Research reference

This project was motivated in part by **t2vec**, described in:

Xiucheng Li, Kaiqi Zhao, Gao Cong, Christian S. Jensen, and Wei Wei. *Deep Representation Learning for Trajectory Similarity Computation*. ICDE 2018. DOI: [10.1109/ICDE.2018.00062](https://doi.org/10.1109/ICDE.2018.00062). An [author-hosted paper](https://xiucheng.org/assets/pdfs/icde18-t2vec.pdf) is available.

DJGtok is an independent implementation. It includes no upstream t2vec source code, pretrained weights, paper figures, or reproduced paper text. It is not an official t2vec continuation or a reproduction of the paper's experiments. The citation acknowledges the research idea, not a software dependency or an endorsement.

## Dependencies

Dependencies are installed separately through Python packaging; no third-party library is vendored in this repository. The core uses NumPy and Pydantic. Optional features use openpyxl or PyTorch. Development tools are listed separately in [pyproject.toml](pyproject.toml).

Each dependency retains its own license and notices. Consult the metadata and license files of the versions you install, particularly when making a bundled application or redistributing dependencies. DJGtok's MIT license does not replace those terms.

## External transit data

The public repository and Python distributions exclude raw and processed Seoul and MBTA feeds, extracted route records, and model checkpoints. Tests and examples use synthetic inputs. Source links and aggregate experimental measurements are included for transparency; they are not a bundled transit dataset.

The Seoul aggregate measurements used Seoul Metropolitan Government's **서울시 버스노선별정류소정보** dataset, snapshot dated **2026-09-02**, obtained from [Seoul Open Data Plaza, OA-1095](https://data.seoul.go.kr/dataList/OA-1095/L/1/datasetView.do). That page identifies the data as **Korea Open Government License, Type 1: attribution**. See the [official Type 1 terms](https://www.kogl.or.kr/info/licenseType1.do). Our processing excluded incomplete route sequences and produced aggregate counts and retrieval measurements. This attribution is retained with those measurements; no Seoul endorsement is implied.

MBTA feeds used during earlier local exploration are also excluded. Anyone obtaining a feed separately should consult the [MBTA GTFS documentation](https://github.com/mbta/gtfs-documentation) and its linked developer agreement.

For the measurement boundary and source identification, see [provenance](docs/provenance.md). For prospective integrations, see [extending to other cities](docs/extending-cities.md). Access conditions and permissions must be checked for the selected provider and intended use; this notice is not a guarantee that every downstream use is permitted.
