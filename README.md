# Ruházati dolgozók produktivitása – MLOps

Az UCI Garment Productivity dataset `actual_productivity` célváltozójának becslése regresszióval.
A cél reprodukálható adatfeldolgozás és egy automatizált, tesztelt MLOps rendszer felépítése.

Az aktuális munka az **1. fázis: fejlesztői környezet, DVC és preprocessing**.
MLflow, REST API, Docker Compose, Evidently/Streamlit, CI és Airflow a későbbi fázisokban készülnek.

A teljes terv és a követelmények megfeleltetése a [megvalósítási tervben](docs/implementation-plan.md)
olvasható. Az elsődleges követelményforrás a [feladatkiírás](docs/assignment.md), a fázisok rendjét
az [implementációs útmutató](docs/implementation-guidelines.md) tartalmazza.

## AI használata

Codex segít a követelmények feldolgozásában, a kód, tesztek és dokumentáció elkészítésében.
A működést tényleges helyi futtatásokkal és automatizált ellenőrzésekkel igazoljuk.

## Fejlesztői környezet

Előfeltételek: Git, Conda, internet a telepítéshez; később Docker és Compose v2.
A támogatott Python-verzió **3.11.14**. A projekt saját környezetet használ, a Conda base nem módosul.

```bash
conda env create --prefix ./.conda/envs/mlops --file environment.yml
conda activate ./.conda/envs/mlops
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
```

A már telepített környezetnél csak az aktiválás szükséges. A VS Code interpreter- és tesztbeállításai
a repositoryban vannak. A `requirements.in` a közvetlen, a `requirements.txt` az összes feloldott
függőség pontos verzióját tartalmazza. A pathspec 0.12.1 a DVC scmrepo-kompatibilitása miatt rögzített.

A Linux x86_64 Conda-csomagok pontos visszaállítása alternatív módon:

```bash
conda create --prefix ./.conda/envs/mlops --file conda-linux-64.lock.txt
conda activate ./.conda/envs/mlops
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
```

## Dataset és verziókezelés

Forrás: [UCI Productivity Prediction of Garment Employees](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees),
[DOI: 10.24432/C51S6D](https://doi.org/10.24432/C51S6D). A mellékelt leírás szerint CC BY 4.0 licencű.
1197 rekord, 15 oszlop, 59 nap, 2015-01-01 és 2015-03-11 között.
Az `actual_productivity` regressziós célváltozó. A `wip` 506 helyen hiányzik;
a 37 darab 1 fölötti célértéket és a tört létszámokat megőrizzük.

A nyers CSV és az eredeti dataset-leíró PDF DVC-verziókezelésben van. Gitbe csak a `.dvc`
pointerfájlok kerülnek. A kezdeti remote a Gitből kizárt `.dvc-remote/` helyi könyvtár.

```bash
dvc pull
dvc push
dvc status
```

Új klónba a helyi remote nem érkezik meg: a kapott forrásadatokat be kell másolni, vagy közös,
hozzáférhető DVC remote-ot kell beállítani. A fejlesztői gépen a DVC belső cache helye a lokális,
Gitből kizárt konfigurációban `.cache/dvc`; más gépen a DVC alapértelmezése használható.
