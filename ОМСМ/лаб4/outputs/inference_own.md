# Інференс на власних прикладах

## 1. Лабораторна робота (EN) (151 слів)

Students of the Institute for Applied System Analysis at Igor Sikorsky Kyiv Polytechnic Institute have compared two transformer models that write short summaries of news articles. The team fine-tuned BART and T5 on twenty thousand stories from the CNN and Daily Mail websites using a single graphics card at home. Each model read the first five hundred tokens of an article and learned to produce three or four sentences that describe the main events. The larger BART model needed more than twice as much time per epoch, but its summaries were closer to the reference highlights written by journalists. The smaller T5 model was faster and lighter, which makes it more convenient for laptops and phones. The students also checked the summaries by hand and noticed that both models mostly copy whole sentences from the beginning of the text. They plan to repeat the experiment with Ukrainian news next semester.

**facebook/bart-base:** Students of the Institute for Applied System Analysis at Igor Sikorsky Kyiv Polytechnic Institute have compared two transformer models that write short summaries of news articles . The students also checked the summaries by hand and noticed that both models mostly copy whole sentences from the beginning of the text . They plan to repeat the experiment with Ukrainian news next semester .

**t5-small:** Students of the Institute for Applied System Analysis at Igor Sikorsky Kyiv Polytechnic Institute fine-tuned BART and T5 on twenty thousand stories. Each model read the first five hundred tokens of an article and learned to produce three or four sentences that describe the main events. The larger BART model needed more than twice as much time per epoch.

## 2. Погода / повінь (EN, вигадані назви) (130 слів)

Heavy rain caused the river Silverbrook to burst its banks on Tuesday, flooding dozens of homes in the small town of Eastmere. Local officials said that more than 300 residents were moved to a school gymnasium and two hotels overnight. No one was seriously injured, but a 70-year-old man was treated for hypothermia after spending several hours on the roof of his car. The town mayor, Linda Hartley, said the water level had risen by almost two metres in less than six hours. Forecasters warned that another 40 millimetres of rain could fall on Thursday, and the regional council has asked people living near the river to prepare emergency bags. Volunteers have already filled more than 5,000 sandbags. Insurance companies estimate the damage at several million pounds.

**facebook/bart-base:** Heavy rain caused the river Silverbrook to burst its banks on Tuesday . More than 300 residents were moved to a school gymnasium and two hotels overnight . A 70-year-old man was treated for hypothermia after spending several hours on the roof of his car .

**t5-small:** More than 300 residents were moved to a school gymnasium and two hotels overnight. No one was seriously injured, but a 70-year-old man was treated for hypothermia after spending several hours on his car roof. Forecasters warn another 40 millimetres of rain could fall on Thursday.

## 3. Спорт (EN, вигадані клуби) (134 слів)

Northbridge Rovers came from behind to beat Eastfield United 3-1 on Saturday and move to the top of the league table for the first time in eleven years. Eastfield took the lead after just six minutes when striker Tom Walsh headed in a corner. Northbridge equalised before half-time through a long-range shot by 18-year-old midfielder Adam Kerr, who was making only his third senior appearance. Two goals in the last fifteen minutes from substitute Marco Silva sealed the victory and sent the home crowd of 24,000 into celebration. Northbridge manager Paul Grant praised his young players and said the team still had a lot of hard work ahead. Eastfield have now lost three matches in a row and their coach admitted that the defence had been too slow.

**facebook/bart-base:** Northbridge Rovers come from behind to beat Eastfield United 3-1 . Eastfield have now lost three matches in a row . Northbridge manager Paul Grant praised his young players and said the team still had a lot of hard work ahead . The home crowd of 24,000 were delighted with the result .

**t5-small:** Northbridge Rovers beat Eastfield United 3-1 on Saturday. Tom Walsh equalised before half-time through a long-range shot by Adam Kerr. Two goals in the last fifteen minutes from substitute Marco Silva sealed the victory and sent home crowd of 24,000 into celebration.

## 4. Технології (EN, вигадана компанія) (120 слів)

A start-up called Voltera has opened a factory that recycles old electric car batteries, recovering lithium, nickel and cobalt that can be used again in new cells. The plant, built on the site of a former steelworks, will process up to 10,000 tonnes of batteries a year and employ about 250 people. Company director Sarah Mills said the process uses water-based chemistry instead of high-temperature furnaces, cutting energy use by almost half. Car makers face new rules that require a growing share of recycled metals in batteries sold from 2030. Environmental groups welcomed the project but said more collection points were needed, because many small batteries from phones and scooters still end up in household waste.

**facebook/bart-base:** Voltera plant will process up to 10,000 tonnes of batteries a year and employ about 250 people . Car makers face new rules that require a growing share of recycled metals in batteries sold from 2030 . Environmental groups welcomed the project but said more collection points were needed, because many small batteries from phones and scooters still end up in household waste .

**t5-small:** Voltera will process up to 10,000 tonnes of batteries a year and employ 250 people. Car makers face new rules that require a growing share of recycled metals in batteries sold from 2030. Environmental groups welcomed the project but said more collection points were needed because small batteries from phones and scooters still end up in household waste.

## 5. Українською (поза мовою навчання) (56 слів)

У Києві студенти КПІ ім. Ігоря Сікорського представили систему, яка автоматично складає короткий зміст новинних статей. Розробники навчили дві нейронні мережі на десятках тисяч англомовних новин і порівняли їх за якістю та швидкістю. За словами авторів, більша модель пише точніші резюме, а менша працює майже вдвічі швидше. Наступним кроком команда планує адаптувати систему до українських новин.

**facebook/bart-base:** У Києві студенти КПІ ім. Ігоря Сікорського представили систему, яка автоматично складає короткий �міст новинних сстратей.

**t5-small:** ор корскоо редставили систему. ка автоматино склада коротки мере на дестка тис анломовни новин.
