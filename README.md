# Predikcia-financnej-hodnoty-nehnutelnosti-pomocou-strojoveho-ucenia

Diplomová práca sa zameriava na vyhotovenie modelu na odhad ceny nehnuteľnosti
pomocou viacerých štrukturálnych, geografických aj environmentálnych charakteristík.
Model, využívajúci voľne dostupné dátové zdroje pre jeho univerzálnosť,
sa snaží priniesť čo najpresnejšie odhady. Počas cesty jeho tvorby je venovaná pozornosť
rôznorodosti dátových zdrojov. Použité modely strojého učenia sú k-NN, Lasso,
Ridge a Elastic net regresia, rozhodovacie stromy, náhodné lesy a XGBoost, s využitím
metrík R2 a RMSE.

Dáta boli ziskané využitím už existujúceho softvéru pre web scraping obsahujúci frameworku Django, ktorý som mierne upravila pre svoje potreby. Ziskavané údaje z online inzerátov boli dalej spracovávane a filtrované pomocou pythonu a následne použite v strojovom učení.
