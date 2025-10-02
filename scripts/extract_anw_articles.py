import json
from pathlib import Path
from xml.etree.ElementTree import Element

from datasets import Dataset
from lxml import etree
from tqdm import tqdm


def get_lemma_definitions(article_element: Element):
    """
    Extract the lemma definitions from an article element, as found in their Kernbetekenis elements.

    Args:
        article_element (Element): The article element.

    Returns:
        dict: The article with its pid, id, and a list of betekenissen, each with their pid, id, lemmavorm, woordsoort, definitie, and minidefinitie.
    """

    artikel = {"pid": int(article_element.attrib["pid"]), "id": int(article_element.attrib["ID"]), "betekenissen": []}
    # Loop over all Kernbetekenis elements
    for kernbetekenis_element in article_element.xpath(".//Kernbetekenis"):
        bet = {
            "pid": int(kernbetekenis_element.attrib["pid"]) if "pid" in kernbetekenis_element.attrib else None,
            "id": kernbetekenis_element.attrib["id"][3:] if "id" in kernbetekenis_element.attrib else None,
            "lemmavorm": None,
            "woordsoort": None,
            "definitie": None,
            "mini_definitie": None,
            "voorbeelden": [],
        }

        woordsoort_el = kernbetekenis_element.xpath(".//Woordsoort/Type")
        if woordsoort_el:
            bet["woordsoort"] = woordsoort_el[0].text

        lemmavorm_el = kernbetekenis_element.xpath(".//Lemmavorm")
        if lemmavorm_el:
            bet["lemmavorm"] = lemmavorm_el[0].text

        def_el = kernbetekenis_element.xpath(".//definitieBody/Definitie")
        if def_el:
            bet["definitie"] = def_el[0].text

        mini_def_el = kernbetekenis_element.xpath(".//definitieBody/MiniDefinitie")
        if mini_def_el:
            bet["mini_definitie"] = mini_def_el[0].text

        voorbeelden = kernbetekenis_element.xpath(".//Voorbeeld/Tekst")
        for voorbeeld in voorbeelden:
            bet["voorbeelden"].append(voorbeeld.text)

        artikel["betekenissen"].append(bet)

    return artikel


def extract_articles_from_xml(fin: str, fout: str | None = None, element_name: str = "artikel"):
    """
    Read articles from a large XML file and write their definitions to a JSONL file.

    Args:
        file_path (str): The path to the XML file.
        element_name (str): The name of the element to focus on (e.g., "artikel").
    """
    pfout = Path(fout) if fout else Path(fin).parent.joinpath("anw_lemma_articles.jsonl")

    with pfout.open(encoding="utf-8", mode="w") as fhout:
        context = etree.iterparse(fin, events=("end",), tag=element_name)
        for _, elem in tqdm(context):
            artikel = get_lemma_definitions(elem)
            fhout.write(json.dumps(artikel) + "\n")
            # Clear the element to free up memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        del context

    return pfout


def explode_in_definitions(artikel_betekenissen: list[dict]):
    defs = {
        "lemmavorm": [],
        "woordsoort": [],
        "definitie": [],
        "mini_definitie": [],
        "voorbeelden": [],
    }
    for betekenissen in artikel_betekenissen:
        for bet in betekenissen:
            voorbeelden = [vb for vb in bet["voorbeelden"] if vb is not None]
            if bet["definitie"] is not None and len(voorbeelden) > 0:
                defs["lemmavorm"].append(bet["lemmavorm"])
                defs["woordsoort"].append(bet["woordsoort"])
                defs["definitie"].append(bet["definitie"])
                defs["mini_definitie"].append(bet["mini_definitie"])
                defs["voorbeelden"].append(voorbeelden)

    return defs


def main(fin: str, num_workers: int | None = None):
    # First extract articles as separate items in the JSONL file
    pfout = extract_articles_from_xml(fin)

    # Then extract the definitions from the articles
    ds = Dataset.from_json(str(pfout))
    print(ds)
    ds = ds.filter(lambda s: len(s["betekenissen"]) > 0, num_proc=num_workers, desc="Filtering empty definitions")
    print("After filtering empty", ds)
    ds_defs = ds.map(
        explode_in_definitions,
        num_proc=num_workers,
        batched=True,
        input_columns="betekenissen",
        remove_columns=ds.column_names,
        desc="Exploding definitions",
    )
    print(ds_defs)

    ds_defs.to_json(str(pfout.parent.joinpath("anw_lemma_definitions.jsonl")))


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        description="Extract lemma articles as well as its definitions from an XML file. Output files will be written to the same directory as the input file."
    )
    cparser.add_argument("fin", type=str, help="Path to the XML file.")
    cparser.add_argument("-j", "--num_workers", type=int, help="Number of workers to use for processing.")
    cargs = cparser.parse_args()
    main(cargs.fin, cargs.num_workers)
