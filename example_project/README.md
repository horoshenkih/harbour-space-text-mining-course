# Project description
## Идея
Реализовать алгоритм text summarization, опираясь на следующую идею:
- каждое предложение представить вектором
- смысл текста = средний вектор (шум сокращается)
- отранжировать предложения по близости к смысловому вектору, взять топ

## Возможная реализация
Сегментацию на предложения и преобразование предложений в векторы взять из spaCy.

## Бейзлайн
Случайные предложения из текста.

## Оценка качества
Data: https://www.tensorflow.org/datasets/catalog/multi_news

Evaluation metric ROUGE: https://rxnlp.com/how-rouge-works-for-evaluation-of-summarization-tasks/

ROUGE-1 F1 = 2 * Precision * Recall / (Precision + Recall)

## Критерии выставления оценок
### 1. max 80 баллов
На 80 баллов достаточно реализовать предложенную идею и показать, что она
лучше бейзлайна.

### 2. +0-10 баллов
Попробовать улучшить предложенный алгоритм.
Для каждого придуманного улучшения нужно посчитать метрику на данных.
За неудавшиеся эксперименты тоже можно получить баллы!

### 3. +0-10 баллов
Оформить алгоритм в виде демо-версии, пригодной к какой-нибудь реальной задаче.

**NB: the project works on Linux only.**

# Installation
1. Download and install Miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
1. Create and activate an empty Conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
1. Install pip (`conda install pip`)
1. Run `./setup.sh`


`streamlit run solution.py 3`
ngrok: https://ngrok.com/
Streamlit + ngrok timeout fix: https://github.com/streamlit/streamlit/issues/443

# Ссылки
- https://github.com/mathsyouth/awesome-text-summarization
- http://nlpprogress.com/english/summarization.html