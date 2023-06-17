# Mte_Consult

[![Status](https://img.shields.io/pypi/status/mte_consult.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/mte_consult)][pypi status]
[![License](https://img.shields.io/pypi/l/mte_consult)][license]

[github]: https://github.com/dthonon/mte_consult/

## Features

- TODO

## Requirements

- TODO

## Installation

Vous pouvez cloner _Mte_Consult_ depuis [github].

## Usage

Il faut d'abord modifier la consultation dans `scrapper/mte/spiders/mte_crawler.py`.
Et ensuite lancer la commande 
`scrapy crawl mte_crawler -o ../data/raw/consultation_xxx.csv:csv`

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [GPL 3.0 license][license],
_Mte_Consult_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/dthonon/mte_consult/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/dthonon/mte_consult/blob/main/LICENSE
[contributor guide]: https://github.com/dthonon/mte_consult/blob/main/CONTRIBUTING.md
[command-line reference]: https://mte_consult.readthedocs.io/en/latest/usage.html
