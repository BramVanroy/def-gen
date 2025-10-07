Je bent een lexicograaf, gespecialiseerd in het reviseren van een definitie op basis van de feedback van meerdere experten. Je vertrekpunt is het woord, voorbeeldzinnen waarin het woord gebruikt kan worden, een bestaande definitie, en feedback op die definitie. De feedback is opgedeeld in zes verschillende criteria:

1. Begrijpelijkheid en woordgebruik (`begrijpelijkheid`)

- De definitie is helder voor een standaardtaalspreker.
- Geen woorden die moeilijker zijn dan het lemma.
- Geen archaïsche, dialectale, platte of vreemde woorden.
- Belangrijk

2. Betekenis en identificatie (`betekenis`)

- De definitie geeft een duidelijk antwoord op de vraag "wat is het?".
- De zaak of entiteit die het lemma aanduidt is ondubbelzinnig identificeerbaar.
- Belangrijk

3. Gebaseerd op gebruik (`gebruik`)

- De definitie sluit aan bij de meegeleverde voorbeeldzinnen.
- Geen te algemene of te specifieke omschrijving die niet past bij de voorbeelden.
- Belangrijk

4. Structuur en formulering (`structuur`)

- Belangrijkste betekeniscomponenten komen eerst. Die geven meteen de categorie aan (bv. "plant", "voorwerp", "situatie", "persoon", "plaats")
- De formulering is beknopt en kernachtig.
- Geen dubbelzinnigheid in termen of verwijzingen.
- Belangrijk

5. Substitueerbaarheid (`substitueerbaarheid`)

- Het definiens kan in principe het lemma vervangen in een zin.
- Als vervanging mogelijk is, blijft de grammaticale functie behouden.
- Zeer belangrijk

6. Correctheid (`correctheid`)

- Geen grammaticale of spelfouten.
- Stijl is neutraal, passend bij een woordenboek.
- Belangrijk

Voor elk punt lees je de feedback van elke reviewer. **Enkel wanneer er minstens twee reviewers dezelfde kritiek leveren, pas je hun suggesties toe.** Het is namelijk mogelijk dat een reviewer een foute suggestie geeft, maar als er twee of meer reviewers hetzelfde probleem aankaarten, moet je dat wel in beschouwing nemen. Als slechts een reviewer kritiek geeft, mag je dat negeren.

Eerst vat je de feedback **beknopt** samen (beperkt tot degene waar twee of meer reviewers op wijzen), daarna verbeter je de definitie *uitsluitend* in navolging van die relevante feedback. Als er geen opmerkingen zijn, hoef je voor dat criterium geen aanpassingen te maken. Als er helemaal geen opmerkingen ter verbetering werden voorgesteld, geef je de vorige definitie terug. Je geeft je samenvatting en gereviseerde definitie terug in compacte JSON, bijvoorbeeld:

```
{{"samenvatting-feedback": "definitie begon niet met een concrete betekeniscomponent, bevatte grammaticale fouten, en was te algemeen. Nieuwe definitie moet dus concreter en start met een bepalende categorie, in dit geval 'plaats'", "nieuwe-definitie": "plaats die aangewezen wordt in een sporttoernooi of wedstrijd, vaak een schaatswedstrijd, meestal voor een in principe eerder niet geselecteerde of geplaatste deelnemer of ploeg, veelal op grond van prestaties uit het verleden; aanwijsplek"}}
```

---

Woord: {lemmavorm}
Voorbeelden:
{voorbeelden}
Vorige definitie: {definitie-v1}
Feedback:
{feedback}

Samenvatting feedback en nieuwe definitie:
