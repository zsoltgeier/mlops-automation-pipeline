# Megvalósítási terv

Az elsődleges követelményforrás a [feladatkiírás](assignment.md). A fejlesztés az
[implementációs útmutató](implementation-guidelines.md) hat fázisában halad, fázisonkénti
ellenőrzéssel. A következő fázis csak külön felhasználói kérésre indul.

Az 1–2. fázis elkészült. A jelenlegi rendszer DVC-adatelőkészítést és MLflow-alapú, tesztelt
modell-életciklust biztosít; a következő feladat a REST API és a Docker Compose integrációja.

## Adatok és szakmai döntések

- Dataset: UCI Productivity Prediction of Garment Employees, 1197 sor és 15 oszlop.
- Célváltozó: `actual_productivity`; regresszió. Elsődleges KPI: MAE, kisebb érték a jobb.
- Megfigyelési időszak: 2015-01-01 – 2015-03-11, 59 különböző nap.
- A `wip` 506 értéke hiányzik. A részlegnevek szóközöket és `sweing` elírást tartalmaznak.
- A 37 darab 1 fölötti célérték és a tört létszámok megmaradnak; ezek nem bizonyított adathibák.
- Időrendi train/validation/test felosztás, teljes napokra számított 60/20/20% körüli aránnyal.
  Az azonos napi megfigyelések együtt maradnak; a maradék napok a test halmazba kerülnek.
- Tanulható imputálás, kódolás és skálázás kizárólag a train halmazon illeszkedik.
- A target nem kerülhet a preprocessing bemenetei közé. A végső test halmaz nem modellválasztási adat.
- A napi túlóra, leállás és ösztönző alapján készülő becslés az adott napi produktivitást magyarázza.
  Napkezdés előtti előrejelzéshez a változók időbeli rendelkezésre állását külön ellenőrizni kellene.

## Egyszerű architektúra

Egy Python-csomag a `src/garments_mlops/` alatt. A DVC kezeli az adatokat és az adatfeldolgozó
pipeline-t. A scikit-learn Pipeline tartalmazza a közös preprocessinget és később a regresszort.
MLflow Tracking és Model Registry biztosítja a modell-életciklust. FastAPI szolgálja ki a
tanítást és predikciót, Streamlit jeleníti meg az Evidently reportokat. Docker Compose indítja
a szolgáltatásokat perzisztens tárolókkal és healthcheckekkel. Airflow a stabil rendszer után kerülhet be.

A champion–challenger összehasonlítás azonos, ismert célértékű validációs adaton dönt promócióról.
A drift monitoring ettől független; önmagában nem válthat modellt. Nincs külön task queue,
Kubernetes vagy általános dependency injection keretrendszer.

## Fázisok és elfogadási feltételek

| Fázis | Tartalom | Bizonyíték |
| --- | --- | --- |
| 1. Adatalapok | Fejlesztői környezet, DVC, tisztítás, split, preprocessing | Unit tesztek, reprodukció és visszaállítás |
| 2. Modell-életciklus | Baseline, challenger, MLflow, Registry, MAE alapú döntés | Valós MLflow runok; gyengébb modell elutasításának tesztje |
| 3. API és Docker | Train/predict/health, naplózás, kezdeti modell, Compose | Egyparancsos indulás és API smoke teszt |
| 4. Monitoring | Evidently, referencia és kérések, Streamlit | Mentett és megjelenített drift report, tesztek |
| 5. Minőség és CI | Coverage, Flake8, Pyright, hookok és Actions | Automatikusan kikényszerített ellenőrzések és image build |
| 6. Beadás és extrák | Airflow, screenshotok, README, PDF és dataset | Valós teljes rendszerellenőrzés és beadási csomag |

## Követelmény–megvalósítás megfeleltetés

| Követelmény | Fázis | Tervezett megvalósítás / ellenőrzés |
| --- | --- | --- |
| Reprodukálható preprocessing és dataset | 1 | `data.py`, `splitting.py`, `preprocessing.py`, `prepare.py`, DVC stage |
| Dataset verziókezelése | 1 | CSV és PDF DVC-pointerei, verziózott feldolgozott kimenet |
| Fejlesztői környezet és pontos csomagverziók | 1–6 | Conda, requirements.txt, VS Code beállítások |
| MLflow Tracking, paraméterek, metrikák és artifactok | 2 | Training runok, dataset-azonosítók, modellek |
| Model Registry és automatikus minőségvédelem | 2 | Champion alias, konfigurálható MAE-feltétel és regressziós teszt |
| Train/predict API és tesztelés | 3 | FastAPI, validáció és integrációs tesztek |
| Kérések és predikciók MLflow-naplózása | 3 | Visszakereshető request/response artifactok |
| Docker és egyparancsos Compose indulás | 3 | Dockerfile, compose.yaml, healthcheckek és volume-ok |
| Evidently és Streamlit | 4 | Külön monitoringmodul, vékony megjelenítő |
| Legalább 95% src coverage; a tesztkód mérése | 1–5 | pytest-cov, branch coverage, külön src és tests küszöb |
| Flake8, 120 karakter, legfeljebb 1% PEP8 hiba | 1–5 | Cél: 0 lint hiba minden Python-fájlban |
| Pyright standard: 0 hiba az src alatt | 1–5 | Külön standard konfiguráció |
| Extra: szelíd strict Pyright | 1–5 | Az assignmentben megadott strict kapcsolók |
| Hookok és GitHub Actions | 5 | Teszt, coverage, lint, típusellenőrzés, Docker build |
| Minimum 50 valódi commit/fő | 1–6 | Fejlesztési lépések utáni, angol Conventional Commits üzenetek |
| Extra: legalább két feature branch, megőrzött history | 1–6 | Foundation és data branch, majd további funkcionális branchek |
| Extra: mainre kizárólag merge | 1–6 | `--no-ff` merge-ek, tiszta munkakönyvtár |
| Extra: Airflow | 6 | Meglévő műveleteket hívó DAG, valós futás és screenshot |
| Extra: bemutató | 6 | Demó forgatókönyv; a személyes bemutatót a hallgató tartja |
| Magyar README és AI használat leírása | 1–6 | Telepítés, működés, döntések és korlátok |
| Valós screenshotok | 6 | MLflow, Registry, REST, Docker, Streamlit/Evidently, Airflow |
| Előfeldolgozott adat és egyoldalas beadási PDF | 1 és 6 | Numerikus dataset, később beadási csomag |

## Korlátok és későbbi beállítások

A helyi DVC remote csak ezen a gépen teszi lehetővé a mentést és visszaállítást. Másik géphez
közös remote vagy a forrásadatok átadása szükséges. A GitHub push, branch protection, távoli CI
és személyes bemutató külön beállítást igényelnek. A személyes beadási adatokat nem találjuk ki.
A commitminimumot valódi, több fázison át tartó fejlesztéssel kell elérni.

## Szakmai háttér

- [scikit-learn: common pitfalls és data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [DVC: pipeline-ok definiálása](https://doc.dvc.org/user-guide/pipelines/defining-pipelines)
- [DVC: pipeline-ok futtatása](https://doc.dvc.org/user-guide/pipelines/running-pipelines)
