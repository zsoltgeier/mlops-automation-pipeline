# 6. fázis – Airflow, végső ellenőrzés és beadás

Ellenőrzés dátuma: 2026-11-09, Europe/Budapest.

## Megvalósítás

A negyedik Compose szolgáltatás a hivatalos, digesttel rögzített Airflow 3.4.0 image-re épül.
A `garment_productivity` DAG vékony SDK-definíció, az üzleti műveletek az `src/` alatti közös
API-kliensben maradnak. A `train_challenger` → `predict_reference` → `monitor_demo` feladatlánc
Random Forest jelöltet tanít, 200 történeti referenciarekordra kér predikciót, majd menti a driftreportot.
A történeti rekordok `source=demo` jelölést kapnak. A predikció az aktuális championt használja;
a modellcsere továbbra is az azonos validációs adaton mért MAE-kapu döntése.

A `POST /monitor` végpont ugyanazt a monitoringműveletet teszi elérhetővé, amelyet a CLI és a
dashboard is használ. A DAG feladateredményei XComban tartalmazzák az API- és MLflow-azonosítókat.
HTTP-hiba esetén a feladat meghiúsul; automatikus retry nincs. A DAG alapból kézzel indítható,
az opcionális `MLOPS_AIRFLOW_SCHEDULE=@daily` UTC szerinti napi ütemezést kapcsol be.
Nincs catchup, egyszerre egy DAG-futás engedélyezett.

Az Airflow külön Python-környezetet használ, csak a függőségek nélküli közös API-kliens és
sémakonstansok kerülnek bele. Ez elkerüli a teljes ML-környezet és az Airflow függőségeinek keverését.
A saját állapotvolume megőrzi a futásokat és naplókat. A healthcheck az adatbázis, scheduler és
DAG processor mellett a DAG REST API-n való megjelenését is ellenőrzi.

## Ellenőrzések

| Ellenőrzés | Eredmény |
| --- | --- |
| `make quality` | Sikeres |
| pytest | 299 sikeres teszt, 0 kihagyás |
| Src statement coverage | 1197/1197, 100% |
| Src branch coverage | 269/270, 99,63% |
| Src együttes coverage | 99,93% |
| Tesztkód statement és branch coverage | 100% |
| Flake8 | 0 hiba az src, tests és dags alatt; 120 karakteres sorlimit |
| Standard és szelíd strict Pyright | Mindkettő 0 hiba és 0 figyelmeztetés |
| `pip check` | Sikeres |
| DVC repro és push | Sikeres; az exportált dataset változatlan |
| Friss Compose indulás üres volume-okkal | Négy egészséges szolgáltatás |
| Izolált teljes rendszerteszt | API, artifactok, Streamlit, két monitoringforrás és Airflow sikeres |
| Izolált környezet takarítása | Saját konténerek, hálózat és négy volume eltávolítva |
| Meglévő bemutatórendszer frissítése | Sikeres, a korábbi modellverziók és naplók megmaradtak |
| Valódi screenshotok | 7 kép a futó rendszer felületeiről és a helyi terminálból |
| Beadási PDF | Egy oldal, beágyazott ékezetes betűkészlet; tartalmi és vizuális ellenőrzés sikeres |
| Beadási CSV és ZIP | 1197 sor, 37 numerikus jellemző, target és split; archive- és hash-ellenőrzés sikeres |

Az új tesztek a referencia-mintavétel szerződését, a hibás API-válaszokat, a monitor végpontot,
az Airflow importhibáit, sikertelen és időtúllépő DAG-okat, valamint a hiányzó vagy hibás
feladatállapotokat is ellenőrzik. Az egyetlen be nem járt src ág továbbra is a Streamlit
`__main__` feltételének importálási ága; a belépési pont és a megjelenítés tesztelt.
Nincs új coverage-kizárás vagy `pragma: no cover`.

A végső bemutatórendszer ellenőrzése 20:06:19 UTC-kor fejeződött be.
Airflow run: `smoke-0fe24d98499c4c07b8b1a038ac77694e`, mindhárom feladat sikeres.
Az izolált, üres volume-os teszt runja: `smoke-83b9bbd2c18f4633807fe608df156273`.
A baseline jelölt validációs MAE-je 0,146764, a championé 0,113125; az új verzió elutasítva,
a champion továbbra is 2. Azonos eredményű Random Forest jelölt sem váltja le.
A géppel olvasható ellenőrzési eredmények és a tesztelt források SHA-256 azonosítója
a helyi, Gitből kizárt `docs/final-results.json` fájlban is megvan.

## Screenshotok és beadási csomag

A [screenshotok](screenshots/) az MLflow Trackinget, a Registryt,
a tényleges Swagger-választ, a Docker CLI kimenetét, a Streamlit/Evidently felületeket és a
sikeres Airflow-futást mutatják. A Docker-kép a helyi terminálban kiadott
`docker compose ps` parancsot és a négy egészséges konténert mutatja.
A drift értékei a kiválasztott demóforgalmat írják le, nem a modell pontosságát.

A helyi `submission/` könyvtárban elkészült az előfeldolgozott CSV, az egyoldalas magyar PDF,
a hash-manifest és a beadási ZIP, benne a hét kép offline másolatával.
A PDF a megadott személyes adatokat, dátumot, repository-linket, screenshotok helyét,
AI-használatot és az ismert korlátokat tartalmazza. A csomag és a személyes adatok Gitből kizártak.
A PDF- és screenshot-generátorok, a saját tesztjeik és kiegészítőik szintén csak helyben maradnak;
nem alkalmazáskódok és nem szükségesek a CI-hez.

## Fennmaradó külső lépések és korlátok

- A GitHub push és a szerveroldali Actions-futtatás még hátravan. A helyi kapuk és az azonos
  Compose smoke script sikeresek, ez nem helyettesíti a távoli CI igazolását. A PDF ezt is feltünteti.
- A DVC remote helyi; új klónban a rögzített hash-t ellenőrző UCI-letöltő állítja helyre a nyers CSV-t.
- A loopback portok, Airflow standalone, SQLite és minden látogatót adminnak tekintő beállítás
  helyi bemutatóra valók. Hálózati telepítéshez külön hitelesítés és infrastruktúra szükséges.
- A Moodle-feltöltés a hallgató feladata.
