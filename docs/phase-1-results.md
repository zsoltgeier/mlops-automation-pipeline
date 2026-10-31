# 1. fázis – Dataset, DVC és preprocessing

Ellenőrzés dátuma: 2026-10-31.

## Elkészült funkciók

- Python 3.11.14 Conda-környezet, pontos csomagverziók, Linux Conda-lock és VS Code beállítások.
- Python-csomag a `src/garments_mlops/` alatt, type hintekkel és angol docstringekkel.
- DVC-verziózott CSV és dataset-leíró PDF, helyi remote és reprodukálható `prepare` stage.
- Stateless validáció, részlegnevek normalizálása, dátumfeldolgozás és duplikációkezelés.
- Teljes napok szerint elkülönített, időrendi train/validation/test halmazok.
- Train-only imputálás, hiányjelzők, skálázás és kategorikus kódolás sklearn Pipeline-ban.
- Ugyanazon tisztítás és preprocessing használata tanítási és predikciós bemenetekre.
- `garments-prepare` CLI, tisztított és numerikus exportok, verziózott trainreferencia.
- `preprocessed.csv` a beadáshoz; paraméter-, séma- és SHA-256 provenance a metadata.json fájlban.
- Unit és integrációs tesztek; külön src és tesztkód coverage-küszöb; `make quality` ellenőrzés.

## Dataset és döntések

A cél az `actual_productivity` regressziós becslése. A 37 darab 1 fölötti célérték és a tört
létszámok megmaradnak. A `wip` 506 hiányát a trainen illesztett imputálás kezeli; az eredeti
adatot nem írjuk felül. A teljes forrásban nincs duplikált megfigyelés.

| Halmaz | Napok | Sorok | Első nap | Utolsó nap |
| --- | ---: | ---: | --- | --- |
| Train | 35 | 711 | 2015-01-01 | 2015-02-10 |
| Validation | 11 | 211 | 2015-02-11 | 2015-02-24 |
| Test | 13 | 275 | 2015-02-25 | 2015-03-11 |

A felosztás seedtől független. A 42-es seed a későbbi tanításhoz van rögzítve. Az azonos napi
megfigyelések együtt maradnak. A test halmaz nem használható modellválasztásra.
A napi működési adatok alapján készülő becslés nem igazolja a napkezdés előtti előrejelzés lehetőségét.

A forrás CSV SHA-256 azonosítója:
`763ffcdd39b7f7f360243c2af8f89ed733e8b5f9e511efeb0c886b83e1b76866`.

## Tényleges ellenőrzések

| Ellenőrzés | Eredmény |
| --- | --- |
| `make quality` a projekt interpreterével | Sikeres |
| pytest | 107 sikeres teszt, warning nélkül |
| Teljes src statement és branch coverage | 100% |
| Tesztkód statement és branch coverage, külön mérve | 100% |
| Összesített coverage | 100%, nincs omit vagy pragma |
| Flake8, 120 karakteres sorlimit | 0 hiba |
| Standard Pyright, csak src | 0 hiba, 0 figyelmeztetés |
| Szelíd strict Pyright, csak src | 0 hiba, 0 figyelmeztetés |
| `pip check` | Nincs hibás függőség |
| requirements telepítési szárazpróba, hálózat nélkül | Minden rögzített csomag telepítve |
| `dvc repro` | 1197 sor, 37 transzformált jellemző, 11 kimeneti fájl |
| `dvc repro --force --no-run-cache` | Mind a 11 kimenet bájtra azonos SHA-256 szerint |
| `dvc push` és `dvc status --cloud` | A cache és a helyi remote szinkronban van |
| Friss Git-klón, üres DVC-cache, `dvc pull` | Mind a 13 adatfájl SHA-256 szerint azonos |
| A klónozott kód tényleges DVC-újrafuttatása | Azonos kimenetek, tiszta Git-munkakönyvtár |

A tesztek a hiányzó és hibás értékekre, a duplikált sémára, a nem alapértelmezett sorindexre,
a rossz paraméterekre, a teljes napok szétválasztására, a train-only statisztikákra, az új kategóriákra,
a teljesen hiányzó train oszlopokra, a szerializálásra, a CLI-re és a valódi datasetre is kiterjednek.

## Verziókezelés

A projektalapok és az adatfeldolgozás külön feature branchben készülnek, az összetartozó kód- és
tesztváltoztatások közös commitokba kerülnek. A mainre a történetet megőrző `--no-ff` merge-ekkel
integráljuk a munkát. A minimum 50 commit/fő követelmény a teljes projekt során teljesítendő.

## Elfogadás és korlátok

Az 1. fázis elfogadási feltételei teljesülnek: az adatok reprodukálhatóan feldolgozhatók,
visszaállíthatók, az automatizált tesztek és minőségellenőrzések sikeresek.

A DVC remote helyi könyvtár. Új géphez a forrásadatok átadása vagy közös remote szükséges;
a klónozási ellenőrzésnél a meglévő helyi remote-ot külön konfiguráltuk.
A klónozott futás ugyanazt a telepített függőségi környezetet használta, a kódot a klónból töltötte be.
Teljesen új gépen történő Conda-telepítést ebben a fázisban nem teszteltünk.

MLflow, modellek, API, Docker Compose, monitoring, CI és Airflow még nem készültek el.
A mostani teszteredmények ezek működését nem igazolják. GitHub push és távoli CI-futás nem történt.
Screenshotok és végső beadási csomag a működő szolgáltatások után készülnek.

A következő fázis a baseline/challenger tanítás és az MLflow modell-életciklus megvalósítása.
