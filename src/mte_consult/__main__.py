"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Mte_Consult."""


if __name__ == "__main__":
    main(prog_name="mte_consult")  # pragma: no cover
