import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_sas("LLCP2020.XPT", format="xport", encoding=None)

print("Broj redova:", df.shape[0])
print("Broj kolona:", df.shape[1])

print(df.head())

df.head(5).to_csv("prvih_5_zapisa.csv", index=False)

pregled = pd.DataFrame({
    "Naziv varijable": df.columns,
    "Tehnicki tip": df.dtypes.astype(str).values,
    "Broj jedinstvenih vrijednosti": df.nunique().values
})

print("\nPregled varijabli:")
print(pregled.to_string(index=False))

pregled.to_csv("pregled_varijabli.csv", index=False)

print("\nGotovo konstantne varijable:")

for kolona in df.columns:
    udjeli = df[kolona].value_counts(normalize=True)

    if not udjeli.empty and udjeli.iloc[0] >= 0.95:
        print(kolona, round(udjeli.iloc[0] * 100, 2), "%")

print("\nBroj ponovljenih identifikatora:")

print(df.duplicated(subset=["_STATE", "SEQNO"]).sum())

nedostajuci = pd.DataFrame({
    "Varijabla": df.columns,
    "Broj nedostajucih": df.isna().sum().values,
    "Postotak nedostajucih": (
        df.isna().mean() * 100
    ).round(2).values
})

nedostajuci = nedostajuci.sort_values(
    by="Postotak nedostajucih",
    ascending=False
)

print("\nPregled nedostajucih vrijednosti:")
print(nedostajuci.to_string(index=False))

nedostajuci.to_csv(
    "nedostajuce_vrijednosti.csv",
    index=False
)

ace_nedostajuci = nedostajuci[
    nedostajuci["Varijabla"].str.startswith("ACE") |
    (nedostajuci["Varijabla"] == "ADDEPEV3")
]

plt.figure(figsize=(12, 6))

stupci = plt.bar(
    ace_nedostajuci["Varijabla"],
    ace_nedostajuci["Postotak nedostajucih"],
    color="steelblue"
)

plt.bar_label(stupci, fmt="%.2f%%", padding=3, fontsize=9)

plt.title("Nedostajuće vrijednosti kod odabranih varijabli")
plt.xlabel("Varijable")
plt.ylabel("Postotak nedostajućih vrijednosti (%)")
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig("grafikon_1.png", dpi=300)
plt.close()

print("\n--- Provjera potpunosti zapisa ---")

print("Potpuno popunjeni redovi:",
      df.notna().all(axis=1).sum())

print("Nepotpuni redovi:",
      df.isna().any(axis=1).sum())

ace_kolone = [c for c in df.columns if c.startswith("ACE")]

odabrane_kolone = ace_kolone + ["ADDEPEV3"]

print("\n--- Potpunost zapisa kod odabranih varijabli ---")

print("Potpuno popunjeni zapisi:",
      df[odabrane_kolone].notna().all(axis=1).sum())

print("Nepotpuni zapisi:",
      df[odabrane_kolone].isna().any(axis=1).sum())

print("Zapisi bez ijednog ACE odgovora:",
      df[ace_kolone].isna().all(axis=1).sum())

print("\n--- Provjera duplikata ---")

broj_duplikata = df.duplicated().sum()

print("Broj potpuno jednakih redova:",
      broj_duplikata)

print("Postotak duplikata:",
      round(broj_duplikata / len(df) * 100, 2), "%")

print("\n--- Provjera posebnih kodova odgovora ---")

for kolona in odabrane_kolone:
    print("\nVarijabla:", kolona)
    print(df[kolona].value_counts(dropna=False).sort_index())

popunjeni = df[odabrane_kolone].notna().all(axis=1)

posebni_kodovi = df[odabrane_kolone].isin([7, 9]).any(axis=1)

valjani = popunjeni & ~posebni_kodovi

print("\n--- Provjera valjanih odgovora ---")

print("Potpuno popunjeni zapisi:",
      popunjeni.sum())

print("Popunjeni zapisi s kodovima 7 ili 9:",
      (popunjeni & posebni_kodovi).sum())

print("Zapisi sa svim valjanim odgovorima:",
      valjani.sum())

print("Udio valjanih zapisa u izvornom datasetu:",
      round(valjani.sum() / len(df) * 100, 2), "%")

print("\n--- Provjera numerickih vrijednosti ---")

numericke_kolone = [
    "_AGE80",
    "SLEPTIM1",
    "MENTHLTH",
    "PHYSHLTH",
    "HTM4",
    "WTKG3",
    "_BMI5"
]

for kolona in numericke_kolone:
    print("\nVarijabla:", kolona)
    print("Minimum:", df[kolona].min())
    print("Maksimum:", df[kolona].max())

print("\n--- Provjera raspona vrijednosti ---")

print("Dob izvan dozvoljenog raspona:",
      (df["_AGE80"].notna() &
       ~df["_AGE80"].between(18, 80)).sum())

for kolona in ["SLEPTIM1", "MENTHLTH", "PHYSHLTH"]:
    print("\nVarijabla:", kolona)
    print(df[kolona].value_counts().sort_index())

print("\n--- Provjera jedinica mjere ---")

print(df[["HEIGHT3", "HTM4", "WEIGHT2",
          "WTKG3", "_BMI5"]].head(10).to_string(index=False))

print("\n--- Provjera datumskih vrijednosti ---")

datumi = pd.to_datetime(
    df["IDATE"].str.decode("utf-8"),
    format="%m%d%Y",
    errors="coerce"
)

print("Neispravni datumi:",
      (df["IDATE"].notna() & datumi.isna()).sum())

print("Najraniji datum:", datumi.min())
print("Najkasniji datum:", datumi.max())

mjeseci = pd.to_numeric(df["IMONTH"], errors="coerce")
dani = pd.to_numeric(df["IDAY"], errors="coerce")
godine = pd.to_numeric(df["IYEAR"], errors="coerce")

neuskladjeni_datumi = datumi.notna() & (
    (datumi.dt.month != mjeseci) |
    (datumi.dt.day != dani) |
    (datumi.dt.year != godine)
)

print("Neuskladjeni datumski zapisi:",
      neuskladjeni_datumi.sum())

print("\n--- Provjera uskladjenosti BMI vrijednosti ---")

visina = df["HTM4"] / 100
tezina = df["WTKG3"] / 100

izracunati_bmi = tezina / (visina ** 2)
evidentirani_bmi = df["_BMI5"] / 100

razlika = (izracunati_bmi - evidentirani_bmi).abs()

print("Broj uporedivih BMI zapisa:",
      razlika.notna().sum())

print("Najveca razlika:", razlika.max())

print("Broj odstupanja vecih od 1:",
      (razlika > 1).sum())

print("\n--- Analiza obrazaca nedostajanja podataka ---")

bez_ace = df[ace_kolone].isna().all(axis=1)

obrasci = pd.DataFrame({
    "Drzava": df["_STATE"],
    "Spol": df["SEXVAR"],
    "Dobna grupa": df["_AGEG5YR"],
    "Bez ACE podataka": bez_ace
})

for grupa in ["Drzava", "Spol", "Dobna grupa"]:
    rezultat = obrasci.groupby(grupa)["Bez ACE podataka"].agg(
        ["count", "sum", "mean"]
    )

    rezultat.columns = [
        "Ukupno ispitanika",
        "Bez ACE podataka",
        "Postotak nedostajucih"
    ]

    rezultat["Postotak nedostajucih"] = (
        rezultat["Postotak nedostajucih"] * 100
    ).round(2)

    print("\nNedostajuci ACE podaci prema:", grupa)
    print(rezultat.to_string())

    rezultat.to_csv(
        "nedostajuci_ACE_" + grupa.replace(" ", "_") + ".csv"
    )

nedostajanje_drzave = (
    obrasci.groupby("Drzava")["Bez ACE podataka"]
    .mean() * 100
)

nazivi_drzava = {
    1: "Alabama",
    4: "Arizona",
    11: "District of Columbia",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    19: "Iowa",
    21: "Kentucky",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    32: "Nevada",
    38: "North Dakota",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    48: "Texas",
    49: "Utah",
    51: "Virginia",
    55: "Wisconsin",
    56: "Wyoming"
}

nedostajanje_drzave = (
    obrasci.groupby("Drzava")["Bez ACE podataka"]
    .mean() * 100
)

potpuno_nedostaju = (nedostajanje_drzave == 100).sum()

prikaz = nedostajanje_drzave[
    nedostajanje_drzave < 100
].copy()

prikaz.index = prikaz.index.map(
    lambda x: nazivi_drzava.get(int(x), str(int(x)))
)

prikaz[f"{potpuno_nedostaju} područje bez ACE podataka"] = 100

prikaz = prikaz.sort_values()

boje = [
    "indianred" if x == 100 else "steelblue"
    for x in prikaz.values
]

plt.figure(figsize=(12, 9))

stupci = plt.barh(
    prikaz.index,
    prikaz.values,
    color=boje
)

plt.bar_label(
    stupci,
    labels=[
        f"{x:.2f}%".replace(".", ",")
        for x in prikaz.values
    ],
    padding=4,
    fontsize=10
)

plt.title(
    "Nedostajanje ACE podataka prema području anketiranja",
    fontsize=13
)

plt.xlabel("Postotak ispitanika bez ACE podataka (%)")
plt.ylabel("Područje anketiranja")
plt.xlim(0, 112)
plt.grid(axis="x", alpha=0.2)
plt.tight_layout()

plt.savefig("grafikon_2.png", dpi=300)
plt.close()




import pandas as pd
import matplotlib.pyplot as plt


def frekvencijska_analiza(df, naziv_varijable, kategorije=None):
    frekvencije = df[naziv_varijable].value_counts(
        dropna=False
    ).sort_index()

    tabela = pd.DataFrame({
        "Kategorija": [
            "Nedostaje" if pd.isna(kod)
            else kategorije.get(kod, str(kod))
            if kategorije is not None
            else str(kod)
            for kod in frekvencije.index
        ],
        "Frekvencija (n)": frekvencije.values,
        "Postotak (%)": (
            frekvencije.values / len(df) * 100
        ).round(2)
    })

    return tabela


def broj_kategorija(df, naziv_varijable):
    return df[naziv_varijable].nunique(dropna=True)


def sacuvaj_frekvencije(tabela, naziv_varijable):
    tabela.to_csv(
        f"frekvencije_{naziv_varijable}.csv",
        index=False,
        encoding="utf-8-sig"
    )


def grafikon_kategorija(
    df,
    varijabla,
    kategorije,
    naslov,
    naziv_datoteke,
    boje="#6594bd",
    oznaka_x="Kategorija"
):
    podaci = df[varijabla]
    valjani = podaci[podaci.isin(kategorije.keys())]

    broj = valjani.value_counts().reindex(
        list(kategorije.keys()),
        fill_value=0
    )

    if broj.sum() == 0:
        print(f"Nema valjanih odgovora za {varijabla}.")
        return

    postotak = broj / broj.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    stupci = ax.bar(
        list(kategorije.values()),
        postotak,
        color=boje
    )

    for stupac, vrijednost in zip(stupci, postotak):
        ax.text(
            stupac.get_x() + stupac.get_width() / 2,
            vrijednost + 1,
            f"{vrijednost:.2f}%",
            ha="center"
        )

    ax.set_title(naslov)
    ax.set_xlabel(oznaka_x)
    ax.set_ylabel("Postotak valjanih odgovora (%)")
    ax.set_ylim(0, max(postotak) + 10)

    plt.tight_layout()
    plt.savefig(
        naziv_datoteke,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


def frekvencije_vise_varijabli(df, varijable):
    rezultati = []

    for naziv in varijable:
        tabela = frekvencijska_analiza(df, naziv)
        tabela.insert(0, "Varijabla", naziv)

        rezultati.append(tabela)

    return pd.concat(
        rezultati,
        ignore_index=True
    )


def pregled_rijetkih_kategorija(sve_frekvencije, prag=1):
    podaci = sve_frekvencije.copy()

    popunjeni = podaci[
        podaci["Kategorija"] != "Nedostaje"
    ].copy()

    ukupno = popunjeni.groupby(
        "Varijabla"
    )["Frekvencija (n)"].transform("sum")

    popunjeni["Postotak popunjenih (%)"] = (
        popunjeni["Frekvencija (n)"] / ukupno * 100
    ).round(2)

    rijetke = popunjeni[
        popunjeni["Postotak popunjenih (%)"] < prag
    ].copy()

    rijetke = rijetke.sort_values(
        "Postotak popunjenih (%)"
    )

    return rijetke


def tabela_kontingencije(
    df,
    varijabla1,
    varijabla2,
    kategorije1,
    kategorije2
):
    podaci = df[
        df[varijabla1].isin(kategorije1.keys())
        & df[varijabla2].isin(kategorije2.keys())
    ][[varijabla1, varijabla2]]

    frekvencije = pd.crosstab(
        podaci[varijabla1],
        podaci[varijabla2]
    ).reindex(
        index=list(kategorije1.keys()),
        columns=list(kategorije2.keys()),
        fill_value=0
    )

    postoci = frekvencije.div(
        frekvencije.sum(axis=1).replace(0, float("nan")),
        axis=0
    ) * 100

    frekvencije = frekvencije.rename(
        index=kategorije1,
        columns=kategorije2
    )

    postoci = postoci.rename(
        index=kategorije1,
        columns=kategorije2
    ).round(2)

    return frekvencije, postoci, len(podaci)


def grafikon_kontingencije(
    postoci,
    kategorija,
    naslov,
    naziv_datoteke
):
    vrijednosti = postoci[kategorija]

    fig, ax = plt.subplots(figsize=(8, 5))

    stupci = ax.bar(
        vrijednosti.index,
        vrijednosti.values,
        color=["#6594bd", "#c98da8"]
    )

    for stupac, vrijednost in zip(stupci, vrijednosti.values):
        ax.text(
            stupac.get_x() + stupac.get_width() / 2,
            vrijednost + 1,
            f"{vrijednost:.2f}%",
            ha="center"
        )

    ax.set_title(naslov)
    ax.set_xlabel("Iskustvo iz djetinjstva")
    ax.set_ylabel("Prijavljena dijagnoza depresije (%)")
    ax.set_ylim(0, max(vrijednosti.values) + 10)

    plt.tight_layout()
    plt.savefig(naziv_datoteke, dpi=300, bbox_inches="tight")
    plt.show()


def viserazinska_kontingencija(
    df,
    varijabla1,
    varijabla2,
    grupna_varijabla,
    kategorije1,
    kategorije2,
    grupe
):
    rezultati = {}

    for kod, naziv in grupe.items():
        podaci_grupe = df[df[grupna_varijabla] == kod]

        frekvencije, postoci, broj_zapisa = tabela_kontingencije(
            podaci_grupe,
            varijabla1,
            varijabla2,
            kategorije1,
            kategorije2
        )

        rezultati[naziv] = {
            "frekvencije": frekvencije,
            "postoci": postoci,
            "broj_zapisa": broj_zapisa
        }

    return rezultati


def grafikon_viserazinske_analize(rezultati, naziv_datoteke):
    import numpy as np

    spolovi = list(rezultati.keys())

    iskustvo_da = [
        rezultati[spol]["postoci"].loc["Da", "Da"]
        for spol in spolovi
    ]

    iskustvo_ne = [
        rezultati[spol]["postoci"].loc["Ne", "Da"]
        for spol in spolovi
    ]

    x = np.arange(len(spolovi))
    sirina = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    stupci1 = ax.bar(
        x - sirina / 2,
        iskustvo_da,
        sirina,
        label="Imali navedeno iskustvo",
        color="#6594bd"
    )

    stupci2 = ax.bar(
        x + sirina / 2,
        iskustvo_ne,
        sirina,
        label="Nisu imali navedeno iskustvo",
        color="#c98da8"
    )

    for stupci in [stupci1, stupci2]:
        for stupac in stupci:
            vrijednost = stupac.get_height()

            ax.text(
                stupac.get_x() + stupac.get_width() / 2,
                vrijednost + 1,
                f"{vrijednost:.2f}%",
                ha="center"
            )

    ax.set_title(
        "Prijavljena dijagnoza depresije prema spolu i iskustvu iz djetinjstva"
    )
    ax.set_xlabel("Spol")
    ax.set_ylabel("Prijavljena dijagnoza depresije (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(spolovi)
    ax.set_ylim(0, max(iskustvo_da + iskustvo_ne) + 12)
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        naziv_datoteke,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


def pregled_numerickih_varijabli(df, klasifikacija):
    maska = klasifikacija["Mjerna ljestvica"].isin([
        "Kvantitativne – diskretne",
        "Kvantitativne – kontinuirane"
    ])

    odabrane = klasifikacija.loc[
        maska,
        ["Varijabla", "Opis (Codebook)", "Mjerna ljestvica"]
    ].copy()

    odabrane = odabrane[
        odabrane["Varijabla"].isin(df.columns)
    ].copy()

    odabrane["Broj popunjenih"] = [
        df[naziv].notna().sum()
        for naziv in odabrane["Varijabla"]
    ]

    odabrane["Broj jedinstvenih vrijednosti"] = [
        df[naziv].nunique(dropna=True)
        for naziv in odabrane["Varijabla"]
    ]

    odabrane["Minimum"] = [
        pd.to_numeric(df[naziv], errors="coerce").min()
        for naziv in odabrane["Varijabla"]
    ]

    odabrane["Maksimum"] = [
        pd.to_numeric(df[naziv], errors="coerce").max()
        for naziv in odabrane["Varijabla"]
    ]

    return odabrane.reset_index(drop=True)


def provjera_numerickih_kodova(df):
    posebni_kodovi = {
        "SLEPTIM1": [77, 99],
        "MENTHLTH": [77, 88, 99],
        "PHYSHLTH": [77, 88, 99],
        "POORHLTH": [77, 88, 99]
    }

    rezultati = []

    for naziv, kodovi in posebni_kodovi.items():
        podaci = pd.to_numeric(df[naziv], errors="coerce")

        for kod in kodovi:
            broj = int((podaci == kod).sum())

            rezultati.append({
                "Varijabla": naziv,
                "Vrijednost": kod,
                "Frekvencija (n)": broj,
                "Postotak popunjenih (%)": round(
                    broj / podaci.notna().sum() * 100, 2
                )
            })

    for naziv in ["NUMMEN", "NUMWOMEN", "DROCDY3_"]:
        podaci = pd.to_numeric(df[naziv], errors="coerce")
        broj = int(((podaci > 0) & (podaci < 0.000001)).sum())

        rezultati.append({
            "Varijabla": naziv,
            "Vrijednost": "Između 0 i 0,000001",
            "Frekvencija (n)": broj,
            "Postotak popunjenih (%)": round(
                broj / podaci.notna().sum() * 100, 2
            )
        })

    return pd.DataFrame(rezultati)


def provjera_vrijednosti_bliskih_nuli(df, varijable):
    rezultati = []

    for naziv in varijable:
        podaci = pd.to_numeric(df[naziv], errors="coerce")

        rezultati.append({
            "Varijabla": naziv,
            "Tačne nule": int((podaci == 0).sum()),
            "Vrijednosti između 0 i 0.000001": int(
                ((podaci > 0) & (podaci < 0.000001)).sum()
            ),
            "Najmanja pozitivna vrijednost": podaci[
                podaci > 0
            ].min(),
            "Najmanja vrijednost iznad 0.000001": podaci[
                podaci >= 0.000001
            ].min()
        })

    return pd.DataFrame(rezultati)


def pregled_najmanjih_vrijednosti(df, varijable):
    rezultati = {}

    for naziv in varijable:
        podaci = pd.to_numeric(
            df[naziv],
            errors="coerce"
        )

        frekvencije = (
            podaci.value_counts(dropna=False)
            .sort_index()
            .head(10)
        )

        rezultati[naziv] = frekvencije

    return rezultati


import numpy as np

def trim_mean(podaci, udio=0.05):
    vrijednosti = np.sort(np.asarray(podaci, dtype=float))
    broj = int(len(vrijednosti) * udio)

    if broj == 0:
        return vrijednosti.mean()

    return vrijednosti[broj:-broj].mean()


def priprema_numerickih_podataka(df):
    varijable = [
        "_AGE80", "SLEPTIM1", "MENTHLTH",
        "PHYSHLTH", "HTM4", "WTKG3", "_BMI5"
    ]

    podaci = df[varijable].copy()

    for naziv in varijable:
        podaci[naziv] = pd.to_numeric(
            podaci[naziv], errors="coerce"
        )

        podaci.loc[
            (podaci[naziv] > 0) &
            (podaci[naziv] < 1e-70),
            naziv
        ] = 0

    podaci["SLEPTIM1"] = podaci["SLEPTIM1"].replace({
        77: np.nan, 99: np.nan
    })

    for naziv in ["MENTHLTH", "PHYSHLTH"]:
        podaci[naziv] = podaci[naziv].replace({
            77: np.nan,
            88: 0,
            99: np.nan
        })

    podaci["WTKG3"] = podaci["WTKG3"] / 100
    podaci["_BMI5"] = podaci["_BMI5"] / 100

    return podaci


def deskriptivna_statistika(df):
    rezultati = []

    for naziv in df.columns:
        podaci = pd.to_numeric(
            df[naziv], errors="coerce"
        ).dropna()

        if podaci.empty:
            continue

        sredina = podaci.mean()
        std = podaci.std()
        q1 = podaci.quantile(0.25)
        q3 = podaci.quantile(0.75)

        cv = (
            std / sredina * 100
            if sredina > 0 and pd.notna(std)
            else np.nan
        )

        rezultati.append({
            "Varijabla": naziv,
            "N": len(podaci),
            "Minimum": podaci.min(),
            "Maksimum": podaci.max(),
            "Sredina": sredina,
            "Standardna devijacija": std,
            "CV (%)": cv,
            "Medijan": podaci.median(),
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1,
            "Asimetričnost": podaci.skew(),
            "Zaobljenost": podaci.kurt(),
            "Skraćena sredina (5%)": trim_mean(
                podaci.to_numpy(), 0.05
            )
        })

    return pd.DataFrame(rezultati).round(2)


def pregled_preostalih_numerickih(df, pregled_numerickih, obradjene):
    preostale = pregled_numerickih[
        ~pregled_numerickih["Varijabla"].isin(obradjene)
    ].copy()

    rezultati = []

    for _, red in preostale.iterrows():
        naziv = red["Varijabla"]
        podaci = pd.to_numeric(df[naziv], errors="coerce")
        frekvencije = podaci.value_counts(dropna=False)

        najcesce = frekvencije.head(5)

        rezultati.append({
            "Varijabla": naziv,
            "Opis": red["Opis (Codebook)"],
            "Mjerna ljestvica": red["Mjerna ljestvica"],
            "Broj popunjenih": int(podaci.notna().sum()),
            "Broj jedinstvenih vrijednosti": int(
                podaci.nunique(dropna=True)
            ),
            "Minimum": podaci.min(),
            "Maksimum": podaci.max(),
            "Najčešće vrijednosti": ", ".join(
                f"{vrijednost}: {broj}"
                for vrijednost, broj in najcesce.items()
            )
        })

    return pd.DataFrame(rezultati)

def pregled_posebnih_numerickih_kodova(df):
    pravila = {
        "CHILDREN": {
            "Nema djece": [88],
            "Odbijen odgovor": [99]
        },
        "DIABAGE3": {
            "Ne znam": [98],
            "Odbijen odgovor": [99]
        },
        "ALCDAY5": {
            "Ne znam": [777],
            "Nije konzumirao alkohol": [888],
            "Odbijen odgovor": [999]
        },
        "DROCDY3_": {
            "Nepoznat ili nedostajući podatak": [900]
        },
        "_DRNKWK1": {
            "Nepoznat ili nedostajući podatak": [99900]
        }
    }

    rezultati = []

    for naziv, kategorije in pravila.items():
        podaci = pd.to_numeric(df[naziv], errors="coerce")

        for opis, kodovi in kategorije.items():
            broj = int(podaci.isin(kodovi).sum())

            rezultati.append({
                "Varijabla": naziv,
                "Značenje": opis,
                "Kodovi": ", ".join(map(str, kodovi)),
                "Frekvencija (n)": broj,
                "Postotak popunjenih (%)": round(
                    broj / podaci.notna().sum() * 100, 2
                ) if podaci.notna().sum() > 0 else float("nan")
            })

    return pd.DataFrame(rezultati)

def priprema_preostalih_numerickih(df):
    varijable = [
        "NUMADULT",
        "NUMMEN",
        "NUMWOMEN",
        "HHADULT",
        "CHILDREN",
        "DIABAGE3"
    ]

    podaci = df[varijable].copy()

    for naziv in varijable:
        podaci[naziv] = pd.to_numeric(
            podaci[naziv],
            errors="coerce"
        )

    for naziv in ["NUMMEN", "NUMWOMEN"]:
        podaci.loc[
            (podaci[naziv] > 0) &
            (podaci[naziv] < 1e-70),
            naziv
        ] = 0

    podaci["HHADULT"] = podaci["HHADULT"].replace({
        77: np.nan,
        99: np.nan
    })

    podaci["CHILDREN"] = podaci["CHILDREN"].replace({
        88: 0,
        99: np.nan
    })

    podaci["DIABAGE3"] = podaci["DIABAGE3"].replace({
        98: np.nan,
        99: np.nan
    })

    return podaci


def priprema_varijabli_alkohol(df):
    varijable = [
        "ALCDAY5",
        "AVEDRNK3",
        "DRNK3GE5",
        "MAXDRNKS"
    ]

    podaci = df[varijable].copy()

    for naziv in varijable:
        podaci[naziv] = pd.to_numeric(
            podaci[naziv],
            errors="coerce"
        )

    alc = podaci["ALCDAY5"].copy()

    sedmicno = alc.between(101, 107)
    mjesecno = alc.between(201, 230)

    podaci["ALCDAY5"] = np.nan

    podaci.loc[sedmicno, "ALCDAY5"] = (
        (alc.loc[sedmicno] - 100) * 30 / 7
    )

    podaci.loc[mjesecno, "ALCDAY5"] = (
        alc.loc[mjesecno] - 200
    )

    podaci.loc[alc == 888, "ALCDAY5"] = 0

    for naziv in ["AVEDRNK3", "DRNK3GE5"]:
        podaci[naziv] = podaci[naziv].replace({
            77: np.nan,
            88: 0,
            99: np.nan
        })

    podaci["MAXDRNKS"] = podaci["MAXDRNKS"].replace({
        77: np.nan,
        99: np.nan
    })

    return podaci


def priprema_varijabli_zdravlje_padovi(df):
    varijable = [
        "POORHLTH",
        "FALL12MN",
        "FALLINJ4"
    ]

    podaci = df[varijable].copy()

    for naziv in varijable:
        podaci[naziv] = pd.to_numeric(
            podaci[naziv],
            errors="coerce"
        )

        podaci[naziv] = podaci[naziv].replace({
            77: np.nan,
            88: 0,
            99: np.nan
        })

    return podaci


def priprema_varijabli_dijabetes(df):
    varijable = [
        "BLDSUGAR",
        "FEETCHK3",
        "DOCTDIAB",
        "CHKHEMO3"
    ]

    podaci = df[varijable].copy()

    for naziv in varijable:
        podaci[naziv] = pd.to_numeric(
            podaci[naziv],
            errors="coerce"
        )

    for naziv in ["BLDSUGAR", "FEETCHK3"]:
        vrijednosti = podaci[naziv].copy()
        rezultat = pd.Series(np.nan, index=podaci.index)

        dnevno = vrijednosti.between(101, 199)
        sedmicno = vrijednosti.between(201, 299)
        mjesecno = vrijednosti.between(301, 399)
        godisnje = vrijednosti.between(401, 499)

        rezultat.loc[dnevno] = (
            vrijednosti.loc[dnevno] - 100
        ) * 30

        rezultat.loc[sedmicno] = (
            vrijednosti.loc[sedmicno] - 200
        ) * 30 / 7

        rezultat.loc[mjesecno] = (
            vrijednosti.loc[mjesecno] - 300
        )

        rezultat.loc[godisnje] = (
            vrijednosti.loc[godisnje] - 400
        ) / 12

        rezultat.loc[vrijednosti == 888] = 0

        podaci[naziv] = rezultat

    podaci["DOCTDIAB"] = podaci["DOCTDIAB"].replace({
        77: np.nan,
        88: 0,
        99: np.nan
    })

    podaci["CHKHEMO3"] = podaci["CHKHEMO3"].replace({
        77: np.nan,
        88: 0,
        98: np.nan,
        99: np.nan
    })

    return podaci


def priprema_varijabli_pusenje(df):
    varijable = [
        "MARIJAN1",
        "LCSFIRST",
        "LCSLAST",
        "LCSNUMCG"
    ]

    podaci = df[varijable].copy()

    for naziv in varijable:
        podaci[naziv] = pd.to_numeric(
            podaci[naziv],
            errors="coerce"
        )

    podaci["MARIJAN1"] = podaci["MARIJAN1"].replace({
        77: np.nan,
        88: 0,
        99: np.nan
    })

   
    for naziv in ["LCSFIRST", "LCSLAST", "LCSNUMCG"]:
        podaci[naziv] = podaci[naziv].replace({
        777: np.nan,
        999: np.nan
    })

    podaci["LCSFIRST"] = podaci["LCSFIRST"].where(
    podaci["LCSFIRST"].between(1, 100)
)
    return podaci


def priprema_zavrsnih_numerickih(df):
    varijable = ["NUMPHON3", "FEETCHK", "CNCRAGE"]

    podaci = df[varijable].copy()

    for naziv in varijable:
        podaci[naziv] = pd.to_numeric(
            podaci[naziv],
            errors="coerce"
        )

    podaci["NUMPHON3"] = podaci["NUMPHON3"].replace({
        7: np.nan,
        8: 0,
        9: np.nan
    })

    podaci["FEETCHK"] = podaci["FEETCHK"].replace({
        77: np.nan,
        88: 0,
        99: np.nan
    })

    podaci["CNCRAGE"] = podaci["CNCRAGE"].replace({
        98: np.nan,
        99: np.nan
    })

    return podaci


def histogram_normalna_krivulja(podaci, naziv, naslov, oznaka_x, datoteka, broj_binova=30):
    import matplotlib.pyplot as plt
    import numpy as np

    vrijednosti = pd.to_numeric(
        podaci[naziv], errors="coerce"
    ).dropna()

    sredina = vrijednosti.mean()
    std = vrijednosti.std()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        vrijednosti,
        bins=broj_binova,
        density=True,
        color="#6594bd",
        edgecolor="white",
        alpha=0.8,
        label="Stvarna raspodjela"
    )

    if std > 0:
        
        x = np.linspace(
    vrijednosti.min(),
    vrijednosti.max(),
    500
)


        y = (
            np.exp(-0.5 * ((x - sredina) / std) ** 2)
            / (std * np.sqrt(2 * np.pi))
        )

        ax.plot(
            x, y,
            color="#c05b80",
            linewidth=2.5,
            label="Normalna krivulja"
        )

    ax.set_title(naslov)
    ax.set_xlabel(oznaka_x)
    ax.set_ylabel("Gustoća")
    ax.set_xlim(vrijednosti.min(), vrijednosti.max())
    ax.legend()

    plt.tight_layout()
    plt.savefig(datoteka, dpi=300, bbox_inches="tight")
    plt.show()


def analiza_ekstremnih_vrijednosti(df, varijable):
    rezultati = []

    for naziv in varijable:
        podaci = pd.to_numeric(
            df[naziv], errors="coerce"
        ).dropna()

        q1 = podaci.quantile(0.25)
        q3 = podaci.quantile(0.75)
        iqr = q3 - q1

        donja = q1 - 1.5 * iqr
        gornja = q3 + 1.5 * iqr

        ekstremi = (podaci < donja) | (podaci > gornja)
        broj = int(ekstremi.sum())

        rezultati.append({
            "Varijabla": naziv,
            "N": len(podaci),
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Donja granica": donja,
            "Gornja granica": gornja,
            "Broj ekstrema": broj,
            "Postotak ekstrema (%)": round(
                broj / len(podaci) * 100, 2
            )
        })

    return pd.DataFrame(rezultati).round(2)


def boxplot_numerickih(df, varijable):
    import matplotlib.pyplot as plt

    for naziv, naslov, datoteka in varijable:
        podaci = pd.to_numeric(
            df[naziv], errors="coerce"
        ).dropna()

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.boxplot(
            podaci,
            vert=False,
            patch_artist=True,
            boxprops={"facecolor": "#6594bd"},
            medianprops={"color": "#c05b80", "linewidth": 2},
            flierprops={
                "marker": ".",
                "markersize": 2,
                "alpha": 0.3
            }
        )

        ax.set_title(naslov)
        ax.set_xlabel("Vrijednost")
        ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(datoteka, dpi=300, bbox_inches="tight")
        plt.show()


def rasprseni_dijagram(df, x_var, y_var, datoteka):
    import matplotlib.pyplot as plt

    podaci = df[[x_var, y_var]].dropna()
    uzorak = podaci.sample(
        n=min(10000, len(podaci)),
        random_state=42
    )

    korelacija = podaci[x_var].corr(podaci[y_var])

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        uzorak[x_var],
        uzorak[y_var],
        alpha=0.2,
        s=8,
        color="#6594bd"
    )

    ax.set_title("Odnos dobi i indeksa tjelesne mase")
    ax.set_xlabel("Dob (godine)")
    ax.set_ylabel("Indeks tjelesne mase (BMI)")

    plt.tight_layout()
    plt.savefig(datoteka, dpi=300, bbox_inches="tight")
    plt.show()

    return len(podaci), korelacija


def boxplot_po_kategoriji(df, numericka, kategorijska, datoteka):
    import matplotlib.pyplot as plt

    podaci = df[[numericka, kategorijska]].copy()
    podaci = podaci[
        podaci[kategorijska].isin([1, 2])
    ].dropna()

    grupe = {
        "Da": podaci.loc[
            podaci[kategorijska] == 1, numericka
        ],
        "Ne": podaci.loc[
            podaci[kategorijska] == 2, numericka
        ]
    }

    statistika = podaci.groupby(kategorijska)[numericka].agg(
        ["count", "mean", "median"]
    )

    statistika.index = statistika.index.map({
        1: "Da",
        2: "Ne"
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        list(grupe.values()),
        tick_labels=list(grupe.keys()),
        patch_artist=True,
        boxprops={"facecolor": "#6594bd"},
        medianprops={"color": "#c05b80", "linewidth": 2},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3}
    )

    ax.set_title("Dani lošeg mentalnog zdravlja prema dijagnozi depresije")
    ax.set_xlabel("Prijavljena dijagnoza depresije")
    ax.set_ylabel("Broj dana")

    plt.tight_layout()
    plt.savefig(datoteka, dpi=300, bbox_inches="tight")
    plt.show()

    return statistika.round(2)
