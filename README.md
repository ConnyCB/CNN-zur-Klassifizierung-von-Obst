**CNN-zur-Klassifizierung-von-Obst**

Um Objekte, wie Obst oder Gemüse, erkennen und klassifizieren zu können, stehen unterschiedliche Machine Learning Frameworks zur Verfügung. Convolutional Neural Networks
(CNN) bieten im Vergleich zu herkömmlichen Neuronalen Netzwerken eine höhere Effizienz bei geringerem Speicher- und Rechenbedarf sowie eine hohe Robustheit und können
daher als State-of-the-Art in der Bild- und Audioverarbeitung angesehen werden (Oldenthal,
2019).
Im Folgenden werden zwei unterschiedliche CNN etabliert und deren Ergebnisse bei der Klassifizierung von Obst gegenübergestellt.

**Initialisierung & Daten**
Die Datenbasis für das Training der Modelle liefert der „Fruits 360“ Datensatz (Oltean,
2017). Er enthält insgesamt 90.483 Fotos von 131 Klassen von Obst- und Gemüsesorten.

Um den Trainingsumfang zu begrenzen, wurden 5 Klassen von Obst (5284 Bilder) ausgewählt, die klassifiziert werden sollen:
- Apple (2134 Bilder)
- Apricot (492 Bilder)
- Limes (490 Bilder)
- Orange (479 Bilder)
- Pear (1689 Bilder)

In Anlehnung an den „Fruits“ Datensatz wurde ein zweites, selbst fotografiertes Test-Datenset entwickelt, um die Modelle mit Fotos anderer Früchte der gleichen Klassen vor eine
realistische Herausforderung zu stellen. 

**Modellierung**

Model 1 (Notebook: EDA_and_CNN_Fruits-Dataset.ipynb) entspricht einem gängigen, auf
Forschung mit ähnlichen Daten basierenden Framework. Es beinhaltet drei ConvolutionalSchichten (32, 64 und 128 Filter) mit dazwischen liegenden Pooling-Schichten (MaxPooling), einem Flatten-Layer und abschließenden Dense-Layern. Als Aktivierungsfunktion
wurde in den Convolutional-Schichten wie auch im letzten Dense-Layer Relu verwendet.
Um die bestmöglichen Gewichte für das CNN zu finden, wurde Adam als Optimierungsfunktion gewählt. Die Verlust-Funktion, also der Fehler zwischen Vorhersage und vorgegebenem Zielwert, wird durch Categorical Crossentropy überwacht.

In Modell 2 (Transfer Learning_Inference_Fruits-Dataset.ipynb) wurde ein vortrainiertes
CNN genutzt, um von den in der Faltungsbasis erlernten Merkmalen zu profitieren und
diese auf die „Fruits“ Daten zu übertragen. Als vortrainiertes Model wurde efficientnet_v2_b0 (Tensor Flow Hub, 2022) ausgewählt, da es im Vergleich zu State-of-the-ArtModellen sehr leichtgewichtig ist und schnelles Training ermöglicht (V. Le & Tan, 2021).
Das Training erfolgte auf der ImageNet Datenbank und somit auf ca. 14 Millionen farbigen
Bildern im RGB-Format. Wie im anderen Modellen wurden auch hier Adam als Optimierungsfunktion und Categorical Crossentropy als Verlustfunktion ausgewählt.
