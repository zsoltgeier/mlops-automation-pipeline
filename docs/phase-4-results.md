# 4. fázis – Evidently és Streamlit

Ellenőrzés dátuma: 2026-11-04, Europe/Budapest.

## Megvalósítás

- Evidently 0.7.23 és Streamlit 1.66.1, a teljes Python-függőséglista pontos verziórögzítésével.
- A DVC-ben kezelt `reference.csv` hash-ellenőrzése; a referencia kizárólag train bemeneteket tartalmaz.
- Current adatok visszaolvasása a sikeres predict runokból, eredeti artifact-hash-ek és runazonosítók megőrzésével.
- Legutóbbi 50 sikeres kérés, legfeljebb 5000 kiválasztott sor; `live` és `demo` források külön kezelése.
- Az API opcionális `source` mezője, az auditban és monitoring CSV-ben is rögzítve.
- Közös stateless tisztítás után 15 jellemző vizsgálata; nincs current adatokon illesztett preprocessing.
- Evidently `DataDriftPreset`, minden jellemzőhöz Jensen–Shannon-távolság, 0,1-es küszöb,
  dataset-szinten legalább 50% eltérő eloszlású oszlop.
- Külön oszloponkénti hiányarányok; teljesen hiányzó oszlophoz nincs kitalált eloszlási pontszám.
- Natív Evidently HTML és JSON, summary, referencia és current CSV mentése MLflow-ba.
- Azonos adat- és beállítás-fingerprinthez korábbi sikeres report újrafelhasználása.
- Streamlit dashboard interaktív reporttal, letöltéssel, oszloppontszámokkal, modellverziókkal,
  tanítási MAE-vel és promóciós eredményekkel; kis mintára és demóra külön jelölés.
- `garments-monitor` CLI és 200 determinisztikus, megjelölt demórekord küldése a valódi predict API-n.
- Harmadik Compose szolgáltatás, API-readiness után induló report-előkészítéssel, saját healthcheckkel.

## Minőségellenőrzés

| Ellenőrzés | Eredmény |
| --- | --- |
| `make quality` | Sikeres |
| pytest | 263 sikeres teszt |
| Src statement coverage | 100% |
| Src branch coverage | 215/216 ág, 99,54% |
| Src statement + branch coverage | 99,92% |
| Tesztkód statement és branch coverage | 100% |
| Összesített coverage | 99,96%, kizárás nélkül |
| Flake8 | 0 hiba, 120 karakteres sorlimit |
| Standard és szelíd strict Pyright | Mindkettő 0 hiba és 0 figyelmeztetés az src alatt |
| `pip check` | Sikeres |
| DVC repro és push | Sikeres; a kimeneti adatok változatlanok, a környezet lockja frissült |
| Compose-konfiguráció és friss image build | Sikeres |

Az egyetlen le nem fedett ág a Streamlit `__main__` feltételének importálási ága; a teljes
dashboard-szkript és annak interakciói lefutnak. Nincs hozzá coverage-kizárás.

Valódi Evidently tesztek vizsgálják az azonos és eltolódott eloszlást, a teljesen hiányzó
oszlopot, valamint a nem véges eredmény elutasítását. A monitoring-integrációs tesztek tényleges
MLflow runokon ellenőrzik a reportartifactokat, a fingerprint szerinti újrafelhasználást,
a források elkülönítését és az artifactmentési hiba `FAILED` státuszát.
A modellverziók és a champion alias változatlanságát külön ellenőrzik.

A Streamlit `AppTest` a tényleges alkalmazást futtatja: automatikus/éles/demó nézet,
frissítés, demóküldés, kis minta, hiányzó adat, hiányzó champion és redaktált backendhiba.
A CLI-tesztek a 200 rekordos kérés reprodukálhatóságát és a valódi HTTP-szerződést is vizsgálják.

Az Evidently legacy teszthelperében lévő `numpy.core` deprekáció egyetlen, modullal és pontos
üzenettel behatárolt pytest-kivételt kapott. A Litestar 2.22.0 a verziólistában rögzített,
Evidentlyvel kompatibilis függőség. A konstans numerikus oszlopok miatt nem definiált
korrelációknál csak az elemzés idejére érvényes NumPy divide/invalid kezelés van; a nem véges
driftpontszám továbbra is hiba. A korábbi MLflow `AnyType` kivétel változatlanul dokumentált.

## Valódi konténeres és böngészős ellenőrzések

A fejlesztői Compose-rendszer meglévő volume-jai megmaradtak. Az API, az MLflow és a dashboard
healthy állapotba került; a korábbi négy modellverzió és a 2-es champion változatlan.
A dashboard a korábbi sikeres API-kérések hat rekordjából reportot készített:

| Adatforrás | Reference/current sorok | Driftarány | Monitoring run |
| --- | --- | --- | --- |
| Nem demóként jelölt API-kérések | 711 / 6 | 10/15, 66,67% | `ddebf25d4bb6413f880775475b25a835` |
| Dashboardból küldött, megjelölt demó | 711 / 200 | 9/15, 60% | `bb5f3e60253d46689a98da6cadb8eba8` |

A hatrekordos ablak szűk és nem reprezentatív minta; a dashboard ezt külön jelzi.
Az eloszláskülönbség nem a modell pontosságának mérése. A demó numerikus értékei szándékosan
módosítottak; a felület ezt jól láthatóan megjelöli, és nem keveri az éles forrással.

Chromiumban a valódi Streamlit alkalmazás és a beágyazott Evidently JavaScript-report is betöltődött.
Megjelentek a driftösszesítők, táblázatok és eloszlási grafikonok; nem történt JavaScript-hiba.
A dashboard demóküldő gombja valódi API-kérést hozott létre, a demó nézetre váltás külön reportot mutatott.
A HTML a dashboardból letölthető, és ugyanaz az MLflow-ban mentett artifact jelenik meg.

## Tiszta telepítés és állapotmegőrzés

Külön ideiglenes Git-klónban, saját projekt- és volume-nevekkel, Conda, feldolgozott adatok és
előzetes runtime nélkül futott a rendszer. Kizárólag a mellékelt nyers CSV került visszamásolásra.
Az egyetlen `docker compose up -d --wait` létrehozta a baseline és Random Forest modelleket,
majd automatikusan 200 megjelölt API-demópredikciót és egy sikeres reportot készített.

- Champion: 2-es verzió, Random Forest, validációs MAE `0.1131253775`.
- Demópredikciós run: `af5c77b7704c414db7d080590183e683`.
- Monitoring run: `53dc946d22a447c0bd729ed34ac6295b`, 711 referencia- és 200 current sor, 60% driftarány.
- Böngészőben automatikusan a DEMÓ jelölésű report jelent meg, JavaScript-hiba nélkül.

A teljes tesztrendszer leállítása és elindítása után ugyanaz a két modellverzió és négy run
maradt meg: két tanítás, egy demópredikció és egy monitoringreport. Nem keletkezett új
demókérés, tanítás vagy report pusztán az újraindítás miatt.

Ezután a korábbi `garments-smoke` teljes HTTP/artifact ellenőrzése is sikeres volt:
health/Swagger/OpenAPI 200, hibás batch 422, három predikció a 2-es modellel,
gyengébb 3-as baseline jelölt elutasítva, champion továbbra is 2. A külön tesztrendszer
konténerei és volume-jai az ellenőrzés után eltávolításra kerültek; a fejlesztői rendszer futva maradt.

A részletes, Gitből kizárt ellenőrzési adatok a helyi `runtime/phase-4-*.json` fájlokban vannak.
A fejlesztői runok az MLflow volume-jában maradnak meg; a tiszta telepítés tesztrunjai a törölt
tesztvolume-okhoz tartoztak. Új klón új runazonosítókat kap. A beadási screenshotok a 6. fázisban készülnek.

## Korlátok és következő fázis

A monitoring az induláskor, illetve felhasználói frissítésre fut; nincs folyamatos háttérütemező.
Az ablak nem teljes forgalmi archívum: 50 sikeres kérésből legfeljebb 5000 kiválasztott sor.
A küszöbök bemutatási alapértékek, üzemi riasztáshoz kalibrálás szükséges.
A teljesen hiányzó oszlopok külön hiányarányt kapnak, és nem részei az eloszlási drift nevezőjének.

A report HTML-jét a saját Evidently-generátor állítja elő; a dashboard nem fogad feltölthető,
tetszőleges HTML-t. A helyi környezet MLflow/Registry tárolója továbbra is megbízható komponens.
A konkurens reportkészítés külön runokat hozhat létre; a fingerprint-újrafelhasználás nem elosztott zár.

A drift nem indít tanítást és nem ír modellt vagy aliast. A modellcserét továbbra is a külön,
ismert célértékű validáción futó champion–challenger kapu kezeli.
A következő fázis a pre-commit hookok és GitHub Actions minőségi ellenőrzések elkészítése.
GitHub push és távoli CI-futtatás nem történt; a minimum 50 commit a teljes projektben teljesítendő.

## Szakmai források

- [Evidently: Dataset, DataDefinition és Report](https://docs.evidentlyai.com/quickstart_ml)
- [Evidently: driftmódszerek és küszöbök](https://docs.evidentlyai.com/metrics/customize_data_drift)
- [Streamlit: interaktív HTML beágyazása](https://docs.streamlit.io/develop/api-reference/text/st.iframe)
