# Rapport - slutprojekt i kursen Pythonprogrammering för AI-utveckling (YKPYT26V2)

## Installation
1. Skapa en virtuell python miljö
pythin venv -m my-env

2. Klona github repositositoriet till en folder. Mappen YKPYT26V2 skapas med alla filer som behövs.

3. Ladda ner datafilen som behövs för att träna modellen
Gå till https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset och klicka på download och följ instruktionerna för att ladda ner filen.
Öppna zip-filen och kopiera "used_cars_data.csv" till mappen "data"
	
4. Kör "python build-linear-regression-model.py" för att bygga modellen

5. Kör "python use-linear-regression-model.py" för att anropa modellen med info om en viss bil och få ut ett uppskattat pris.


## Introduktion
Projektet går ut på att välja ett existerande problem eller frågeställning att lösa eller utgå från existerande data och analysera den för att hitta intressant information och sen bygga en process för att analysera datat och bygga och använda en machine-learning modell eller neuralt nätverk som löser problemet eller besvara en viss fråga.
Efter att ha analyserat två olika typer av data (aktiekurser och pris på begagnade bilar) så valde jag att gå vidare med att analyser priset på begagnade bilar för att försöka besvara frågan "Vilket försäljningspris kan jag förvänta mig att få för min bil".

## Dataset
https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset
Dataset:et innehåller ca 9 GB av information från den amerikanska marknaden, vilket gör att informationen inte nödvändigtvis går att använda för den svenska marknaden.
Delproblem #1: Den stora mängden data gör att det vi behöver ta fram en process som gör att vi kan snabba upp inläsningen av data vid många upprepade exekveringar.

Först analyserades datat och de kolumner som är tillgängliga. Datat läses in vilket tar ca 90 sekunder. Alternativet är att dela upp datat i mindre filer under utvecklingsprocessen eller att hitta ett sätt att snabba upp inläsningen genom cache-hantering. Genom att spara ner dataset i ett binärt format (df.to_pickle()) snabbades inläsningen upp till ca 5 sekunder, vilket är tillräckligt snabbt för mig för att inte dela upp datat ytterligare.

Korrelationsdiagram (scatter-diagram) skapades för "Årsmodell vs Pris" och "KördaMil vs Pris". Rimligtvis borde båda ha en stor inverkan på slutpriset men även ha en stor korrelation mellan sig och därmed kanske bara en av dem behöver tas med i modellen. Skapade därför också ett korrelationsdiagram för "KördaMil vs Årsmodell" som visar korellationen som är stor. 

Utförde en rensning av rader där det saknades information

Valde ut följande kolumner för att skapa en första enkel model för att få klart uppgiften i tid:
- Märke
- Modell
- Årsmodell
- VaritMedOmOlycka
- KördaMil
- Pris

## Modell
Modellen ska kunna uppskatta en bils pris utifrån ett antal kända parametrar.
- Typen av modell för att uppskatta ett kontinuerlig numeriskt värde kallas för Regressionsmodell.
Det finns många typer av regressionsmodellen men i den här kursen är inte fokus på att testa många modeller utan att förstå processen så därför väljer jag den enklaste modellen, Linjär Regressionsmodell som anpassad att ta fram kontinuerliga värden där relationerna är linjära.
(Har noterat ett icke-linjärt samband mellan pris och årsmodell. Om man kollar grafer så ser det ut att vara någon form av exponentiellt samband men det lämnar jag att utreda om det finns tid.)

Två tester gjordes

- Försök 1 tränades och testades med data från en enda modell och märke (Volvo XC60)
Modellen testades på 2795 och tränades på 8370 bilförsäljningar
R^2 blev då 78,68. %

- Försök 2 tränades och testades med data från samtliga märken och modeller av bilar.
Modellen testades på 1839495 och tränades på 5518475 bilförsäljningar
R^2 blev då 18.78% %

Modellen sparades för att kunna återanvändas. Ett exempel på gjordes också i en notebook som finns här:
https://github.com/forslundaren/YKPYT26V2/tree/main#:~:text=2%20hours%20ago-,YKPYT26V2.ipynb,-Skapades%20med%20Colab

Det verkar som vi har ett problem här som behöver utredas.
Problemet består av att förhållandet mellan olika features inte är samma för olika märken och modeller. Det går alltså inte att, på ett enkelt sätt, skapa en enda modell för alla typer av bilar.
- En lösning är att skapa en modell för varje märke, eller i värsta fall för varje märke och modell. 
- En annan möjlig lösning som vore intressant att testa är att bygga ett neuralt nätverk och se om den kan hantera alla olika märken och modeller för att uppskatta försäljningspris.

För att höja kvaliten på uppskattningen så borde man kunna inkludera fler features. 
