# behavior_v1: проверка

Окон в обучении: 1291, в проверке: 745 (группы: kth-person02, kth-person03, kth-person06, kth-person14, kth-person20, kth-person23, kth-person25, urfall-adl-11, urfall-adl-17, urfall-adl-20, urfall-adl-23, urfall-adl-25, urfall-adl-26, urfall-adl-27, urfall-adl-30, urfall-adl-34, urfall-adl-37, urfall-adl-39, urfall-fall-09, urfall-fall-17, urfall-fall-20, urfall-fall-30). Эпох: 34.
Macro-F1 на проверке: **0.726**.

| Класс | Окон | Precision | Recall |
|---|---|---|---|
| normal | 512 | 92% | 96% |
| run | 225 | 97% | 81% |
| fall | 8 | 25% | 62% |
| lift_back | 0 | — | — |

Матрица ошибок (строки — правда, столбцы — ответ модели):

| | normal | run | fall | lift_back |
|---|---|---|---|---|
| **normal** | 491 | 6 | 15 | 0 |
| **run** | 42 | 183 | 0 | 0 |
| **fall** | 3 | 0 | 5 | 0 |
| **lift_back** | 0 | 0 | 0 | 0 |
