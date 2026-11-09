# 5. fázis – Minőségi kapuk, Git hookok és GitHub Actions

Ellenőrzés dátuma: 2026-11-09, Europe/Budapest.

## Megvalósítás

- Pre-commit 4.7.0 és tranzitív függőségei rögzítve a közös requirements listában.
- Telepített `pre-commit` és `pre-push` Git hookok, verziózott helyi konfigurációval.
- Commitkor teljes minőségi kapu: Flake8, standard és szelíd strict Pyright, pytest,
  külön src/tests coverage-küszöb és `pip check`. Nincs fájlnévszűréssel kimaradó Python-modul.
- Push előtt Compose-validáció és valódi image build, időkorlátos Docker-parancsokkal.
- Hordozható Makefile: a kiválasztott Python interpretert a Pyright és a DVC alfolyamatai is használják.
- JUnit és coverage XML a géppel olvasható teszteredményekhez.
- GitHub Actions workflow push, pull request és kézi indítás eseményekre, két egymásra épülő jobbal.
- Python 3.11.14, Node.js 22 a CI-ben; teljes commitazonosítóval rögzített hivatalos Actionök,
  csak olvasási repository-jog és párhuzamos branch-futtatások megszakítása.
- Hash-ellenőrzött UCI-adatletöltő, amely tiszta klónban és CI-ben is azonos nyers CSV-t állít helyre.
- Kiterjesztett smoke teszt Streamlit-eléréssel és mindkét monitoringforrás tényleges artifactellenőrzésével.
- Önálló, izolált Compose smoke script: egyedi projekt, dinamikus portok, üres volume-ok,
  diagnosztikai naplók és kilépéskori takarítás.

A `quality` job ugyanazt a pre-commit hookot indítja, mint a helyi commit. A Docker job csak
annak sikere után épít és tesztel. Az XML-eket és konténeres naplókat a workflow siker és hiba
esetén is feltölti, ha létrejöttek; az artifactok megőrzési ideje 14 nap.
Nincs sikeresnek álcázott ellenőrzés vagy `continue-on-error` a minőségi lépésekben.

## Tesztek és számszerű eredmények

| Ellenőrzés | Eredmény |
| --- | --- |
| Teljes `make quality` | Sikeres |
| pytest | 281 sikeres teszt |
| Src statement coverage | 100% |
| Src branch coverage | 239/240 ág, 99,58% |
| Src statement + branch coverage | 99,93% |
| Tesztkód statement és branch coverage | 100% |
| Összesített coverage | 99,96%, kizárás nélkül |
| Flake8 | 0 hiba, 120 karakteres sorlimit |
| Standard és szelíd strict Pyright | Mindkettő 0 hiba és 0 figyelmeztetés |
| `pip check` | Sikeres |
| Pre-commit konfiguráció validálása | Sikeres |
| Valódi Git commit a telepített quality hookkal | Sikeres |
| Pre-push Docker hook sikeres ága | Sikeres image build |
| Actionlint 1.7.12 | Workflow hibamentes |
| Bash szintaxisellenőrzés és tényleges scriptfuttatás | Sikeres |
| DVC repro és push | Sikeres; a feldolgozott adatok változatlanok |

A forrás egyetlen nem bejárt ága továbbra is a Streamlit `__main__` feltételének importálási
ága. A belépési pont és a dashboard teljes megjelenítése/interakciói tesztelve vannak.
Az új letöltő tesztjei a helyes és eltérő hash-t, meglévő fájl védelmét, túlméretes ZIP/CSV-t,
hálózati és írási hibát, valamint a standalone CI-belépési pontot is ellenőrzik.

A monitoring smoke negatív tesztjei hibás HTTP-státuszt, nem befejezett runokat, hibás natív
reportot, eltérő sorszámot, adatforrást, modellverziót és mentett summaryt utasítanak el.
A korábbi API-, Registry-, promóciós és drift-integrációs tesztek is sikeresek maradtak.

## A kapuk tényleges elutasító működése

Külön ideiglenes klónban, a projekt `.conda` könyvtára nélkül, más útvonalú standard venvvel
futott az interpreter- és hookellenőrzés. A venv a már telepített csomagokat is elérte;
a független, teljes függőségtelepítés a friss Docker-buildben történt.
Az UCI-letöltés és mindkét Pyright mód ebben a klónban is sikeres volt.

| Szándékos hiba az izolált ellenőrzésben | Tényleges eredmény |
| --- | --- |
| F401 lint-hiba egy stage-elt Python-fájlban | A Git commit sikertelen, a hook megállítja |
| Hibás típus-hozzárendelés | A Git commit sikertelen, Pyright `reportAssignmentType` |
| Nem létező Docker build input | A pre-push hook sikertelen |
| Src 76%, tesztkód 100%, együttes coverage kb. 98% | Az összesített kapu átmenne, a külön src-kapu hibázik |
| Src 100%, tesztkód 76%, együttes coverage kb. 98% | Az összesített kapu átmenne, a külön tesztkód-kapu hibázik |

A hibás próbák a külön klónban és ideiglenes coverage-fixture-ökben futottak. Nem kerültek
a projekt Git-történetébe vagy termékkódjába. Eredményük a helyi
`runtime/phase-5-gates/results.json` és a kapcsolódó naplók alatt található.

## Valódi teljes rendszerellenőrzés

A `make system-smoke` új image-et épített a rögzített requirements listából, majd saját
`garments-check-*` projektben indította az MLflow-t, API-t és dashboardot. Mindhárom healthy lett.
A hostportokat a Docker választotta, ezért a fejlesztői portokkal nem ütközött.

| Művelet | Eredmény |
| --- | --- |
| Kezdeti baseline és Random Forest | Automatikusan elkészültek, champion 2 |
| Health, Swagger, OpenAPI | HTTP 200 |
| Háromrekordos predikció | A 2-es modellverzióval sikeres |
| Baseline tanítás | Új 3-as verzió, gyengébb jelölt elutasítva, champion 2 |
| Üres predikciós batch | HTTP 422 |
| Request/response és current artifactok | HTTP-n letöltve, tartalmuk ellenőrizve |
| Streamlit kezdőlap és health | HTTP 200 |
| Live report | 711 referencia, 3 current sor, 66,67% driftarány |
| Megjelölt demo report | 711 referencia, 200 current sor, 60% driftarány |
| Natív Evidently HTML/JSON, mentett summary, sorszámok és verziók | Mindkét forrásnál ellenőrizve |
| Kilépéskori takarítás | A teszt konténerei, hálózata és három volume-ja eltávolítva |

A smoke eredményének ideje: `2026-11-09T18:48:05.927887+00:00` (UTC).
Live monitoring run: `d16081a1c2fc4f9991374e37faa4c3f2`;
demo monitoring run: `fb56c0a9339c4325b98a9f8b06bae801`.
Ezek a teszthez létrehozott és a takarításkor törölt volume-okhoz tartoztak.
A teljes eredmény a Gitből kizárt
`runtime/system-smoke/garments-check-1794249956-18890/smoke.json` fájlban maradt meg,
a build-, ellenőrzési, szolgáltatás- és takarítási naplókkal együtt.

A háromrekordos live minta nem reprezentatív, a jelzett drift nem modellpontossági eredmény.
A demó külön jelölt, mesterségesen eltolódott bemenetekből áll. A monitoring nem indított modellcserét.
A fejlesztői projekt három konténere leállítva maradt; a modell- és reportvolume-jai megmaradtak.

## Működési korlátok és következő fázis

A workflow definiált, actionlinttel ellenőrzött; a benne használt helyi minőségi kapu és teljes
Docker-ellenőrzés valóban lefutott. GitHub Actions szerveroldali futás nem történt, mert push sem volt.
Az első GitHub-futtatás eredményét a push után külön kell ellenőrizni. A branch protection és
a kötelező státuszellenőrzések GitHub-beállítása még hátravan.

A nyers CSV helyreállításához internet és az UCI elérhetősége kell. Az eredeti dataset-leíró PDF
továbbra is a mellékletből vagy DVC-ből állítható helyre. A script és a build hook GNU `timeout`
parancsot használ, a támogatott helyi környezet Linux/WSL, a CI Ubuntu 24.04.
A hookot minden klónban telepíteni kell; a verziózott konfiguráció ezt önmagában nem végzi el.

A Python-kapuk nem támaszkodnak Dockerre; a build hook és a rendszerellenőrzés futó Engine-et igényel.
Minden hosszabb Docker-művelet és a takarítás időkorlátos. A GitHub job timeoutja vagy a runner
kényszerített megszüntetése esetén a shell trap lefutása nem garantált; az ideiglenes runner környezete megszűnik.

Az 5. fázis helyben ellenőrizhető elfogadási feltételei teljesültek. Airflow, végső screenshotok,
beadási PDF/ZIP és a teljes projekt végső auditja a 6. fázis feladata. A minimum 50 valódi commit
követelménye a teljes projekt végére teljesítendő.

## Szakmai források

- [Pre-commit: hookok és telepítés](https://pre-commit.com/)
- [GitHub Actions: workflow szintaxis](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
- [GitHub Actions: artifactok](https://docs.github.com/en/actions/tutorials/store-and-share-data)
- [UCI: Productivity Prediction of Garment Employees](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees)
